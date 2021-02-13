// -*- coding:utf-8 -*-
//
// C++11 implementation of Efficient K-NN Graph Construction Algorithm NN-Descent:
//    Wei Dong et al., "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures", WWW11
//    ( http://www.cs.princeton.edu/cass/papers/www11.pdf )
//
// Author: Tatsuya Shirakawa
//
// Implementation Notices
// - some additional joins are added (these are not in paper)
//   - join the center node to the its nbd nodes.
//   - random join ( join random #num_random_join node )
//   - randomly break the tie nodes ( w.p. perturb_rate )

#pragma once
#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <thread>

#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <matplotlibcpp.h>

namespace nng
{
  namespace plt = matplotlibcpp;
  template <class PointT, class SimT>
  struct NNDescent
  {

  public:
    typedef PointT point_type;

    typedef SimT sim_type;

    typedef typename sim_type::value_type sim_value_type;

    struct node_type
    {
      node_type(const point_type &data) : data(data) {}
      node_type() : data() {}
      point_type data;
    };

    struct nbd_elem_type
    {
      nbd_elem_type() : node_id(), similarity(std::numeric_limits<sim_value_type>::min()) {}
      nbd_elem_type(const std::size_t node_id, const sim_value_type similarity) : node_id(node_id), similarity(similarity) {}
      std::size_t node_id;
      sim_value_type similarity;
    };

    struct comp_similarity
    {
      inline bool operator()(const nbd_elem_type &e1, const nbd_elem_type &e2) const
      {
        return e1.similarity > e2.similarity;
      }
    };

    typedef std::vector<nbd_elem_type> nbd_type;

    struct nearer
    {
      inline bool operator()(const nbd_elem_type &elem1, const nbd_elem_type &elem2) const
      {
        return elem1.similarity > elem2.similarity;
      }
    };

    typedef std::mt19937 rand_type;

  public:
    NNDescent(const uint32_t K, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0), const sim_type &sim = sim_type())
        : K(K), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nodes(), nbds(), rand(rand), sim(sim), checked()
    {
    }

    NNDescent(const std::size_t K, const std::vector<point_type> &datas, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0), const sim_type &sim = sim_type())
        : K(K), nodes(), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nbds(), rand(rand), sim(sim), checked()
    {
      init_graph(datas);
    }

  public:
    void init_graph(const std::vector<point_type> &datas)
    {
      if (datas.size() == 0)
      {
        return;
      }

      // clear
      this->clear();

      // set datas
      for (const point_type &data : datas)
      {
        this->nodes.push_back(node_type(data));
      }

      // construct random neighborhoods
      this->init_random_nbds();

      drawGraph("Init Graph", "init_graph.jpg");
    }

    std::size_t update()
    {
      if (this->nodes.size() < 2 || this->K == 0)
      {
        return 0;
      } // cannot create nbd

      const std::size_t N = this->nodes.size();

      std::vector<nbd_type> old_nbds(N), new_nbds(N), old_rnbds(N), new_rnbds(N);

      // Process 1
      // set old_nbds / new_nbds
      for (std::size_t i = 0; i < N; ++i)
      { // ToDo: parallel for i
        this->prep_nbd(i, new_nbds[i], old_nbds[i]);
      }

      // Process 2
      // set old_rnbds / new_rnbds, 生成反向邻居。
      for (std::size_t i = 0; i < N; ++i)
      {
        const auto &new_nbd(new_nbds[i]), &old_nbd(old_nbds[i]);
        for (const nbd_elem_type &elem : old_nbd)
        {
          assert(elem.node_id != i);
          old_rnbds[elem.node_id].push_back(nbd_elem_type(i, elem.similarity));
        }
        for (const nbd_elem_type &elem : new_nbd)
        {
          assert(elem.node_id != i);
          new_rnbds[elem.node_id].push_back(nbd_elem_type(i, elem.similarity));
        }
      }

      // Process 3
      // local join
      std::size_t update_count = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (std::size_t i = 0; i < N; ++i)
      { // ToDo: parallel for i
        update_count += this->local_join(i, new_nbds[i], old_nbds[i], new_rnbds[i], old_rnbds[i]);
      }

      return update_count;
    }

    void operator()(const std::vector<point_type> &datas, const std::size_t &max_epoch = 100, const double &delta = 0.001)
    {
      this->operator()(this->K, datas, max_epoch, delta);
    }

    /*
     * 这个重载符主要用于构建index。而不是用于查询。
     * 参数:
     *  delta: early termination的参数。
     *  max_epoch: 原论文没有，是本实现加的。也是为了如果没达到delta*KN，但是到了max_epoch，也可以提前退出。
     */
    void operator()(const std::size_t K, const std::vector<point_type> &datas, const std::size_t &max_epoch = 100, const double &delta = 0.001)
    {
      this->K = K;

      // initialize graph
      std::cout << "init graph ... " << std::flush;
      this->init_graph(datas);
      std::cout << "done" << std::endl;

      // Update
      for (std::size_t epoch = 0; epoch < max_epoch; ++epoch)
      {
        std::cout << "update [" << epoch + 1 << "/" << max_epoch << "] ..." << std::flush;
        std::size_t update_count = update();
        std::size_t KN = this->rho * this->K * this->nodes.size();
        std::cout << " " << update_count << "/" << KN << std::endl;

        drawGraph(std::string("Iteration") + std::to_string(epoch + 1), std::string("Iteration") + std::to_string(epoch + 1) + std::string(".jpg"));
        if (update_count <= delta * KN)
        {
          break;
        }
      }
    }

  private:
    void clear()
    {
      this->nodes.clear();
      this->nbds.clear();
      this->checked.clear();
    }

    // 随机配对每个节点的邻居
    void init_random_nbds()
    {
      const auto N = this->nodes.size();

      //      const uint32_t K2 = (K <= N-1)? K : static_cast<uint32_t>(N-1);
      uint32_t K2 = this->K;
      if (K2 > N - 1)
      {
        K2 = N - 1;
      }
      assert(K2 <= this->K && K2 <= N - 1);
      this->nbds.clear();
      this->nbds.resize(N);
      this->checked.clear();
      this->checked.resize(N);

      if (N < 2 || this->K == 0)
      {
        return;
      } // cannot create nbd

      for (std::size_t i = 0; i < N; ++i)
      { // ToDo: parallel for i

        std::unordered_set<std::size_t> chosen;

        // assumed K << N
        for (std::size_t j = 0; j < K2; ++j)
        {
          std::size_t n = rand() % (N - 1);
          if (n >= i)
          {
            ++n;
          }
          assert(i != n);
          chosen.insert(n);
        }
        assert(chosen.size() <= K2);
        // set neighborhood
        const auto &node(this->nodes[i]);
        auto &nbd(this->nbds[i]); //这是一个vector,存储每个节点i对应的邻居。
        nbd.resize(chosen.size());
        std::size_t count = 0;
        for (auto j : chosen)
        {
          assert(i != j);
          nbd[count++] = nbd_elem_type(j, this->sim(node.data, this->nodes[j].data));
        }
        //按similarity降序排列
        std::sort(nbd.begin(), nbd.end(), nearer());

        this->checked[i].resize(nbd.size(), false); //每个节点的邻居默认checked都设成false，和原论文是反的，原论文默认是true。

        assert(nbd.size() > 0); // because K > 0 and N > 1
        assert(nbd.size() <= this->K);
      }
    }

    typedef enum
    {
      NOT_INSERTED,
      INSERTED
    } join_result_type;

    std::size_t compute_ub(const nbd_type &nbd, const std::size_t joiner,
                           const sim_value_type s, const std::size_t K2) const
    {
      assert(K2 > 0);
      if (nbd.back().similarity >= s)
      {
        return K2;
      }
      else
      {
        return std::upper_bound(nbd.begin(), nbd.end(), nbd_elem_type(joiner, s), comp_similarity()) - nbd.begin();
      }
    }

    void drawGraph(std::string title, std::string filename)
    {
      const auto N = this->nodes.size();
      point_type x;
      point_type y;

      for (std::size_t i = 0; i < N; ++i)
      {
        const auto &node(this->nodes[i]);
        const auto DIM = node.data.size();
        for (std::size_t j = 0; j < DIM; j++)
        {
          const auto v = node.data[j];
          (j % DIM == 0) ? x.push_back(v) : y.push_back(v);
        }
      }
      plt::figure_size(1200, 1200);
      plt::scatter(x, y);

      for (std::size_t i = 0; i < N; ++i)
      {
        point_type xnbd;
        point_type ynbd;
        auto &nbd(this->nbds[i]); //这是一个vector,存储每个节点i对应的邻居。
        //std::cout << nbd.size() << "," << this->K << "\n";
        for (std::size_t j = 0; j < nbd.size(); j++)
        {
          xnbd.push_back(x[i]);
          ynbd.push_back(y[i]);
          nbd_elem_type nb = nbd[j];
          const auto &node(this->nodes[nb.node_id]);
          const auto DIM = node.data.size();
          for (std::size_t k = 0; k < DIM; k++)
          {
            const auto v = node.data[k];
            (k % DIM == 0) ? xnbd.push_back(v) : ynbd.push_back(v);
          }
        }
        plt::plot(xnbd, ynbd);
      }

            // Set x-axis to interval [0,1000000]
      plt::xlim(0.0, 1.0);
      plt::ylim(0.0, 1.0);

      // Add graph title
      plt::title(title);
      // Enable legend.
      //plt::legend();
      plt::save(filename);
      //plt::xlabel('Numbers',fontsize=14)
      plt::close();
    }

    // 把joiner和base连接起来，也就是把joiner放入到base的邻居里
    join_result_type join(const std::size_t base, const std::size_t joiner)
    {
      assert(base != joiner);
      assert(this->nodes.size() > 1 && this->K > 0);
      const auto &base_node(this->nodes[base]), &joiner_node(this->nodes[joiner]);
      auto &nbd(this->nbds[base]);
      auto &chkd(this->checked[base]);
      assert(nbd.size() == chkd.size());
      const auto s = this->sim(base_node.data, joiner_node.data);
      const auto K2 = this->nbds[base].size();

      // 如果相似度小于最小的相似度，也就是说距离大于最大的距离，就不连接了
      if (s < nbd.back().similarity && K2 == this->K)
      {
        return NOT_INSERTED;
      }

      const nbd_elem_type joiner_elem(joiner, s);
      assert(K2 > 0);
      assert(K2 <= this->K && K2 <= this->nodes.size() - 1);

      // find the position i such that nbd[i] is where joiner will be inserted

      // search ub
      const std::size_t ub = std::upper_bound(nbd.begin(), nbd.end(), joiner_elem, comp_similarity()) - nbd.begin();

      // to prevent perturbation of nbd, the probability of replacement
      // with most dissimilar element in nbd shoule be suppressed
      //      if(ub == K2 && (rand() % B) >= prB){
      //      const auto B = 1000000; // Big Integer (ad-hoc)
      const auto SHIFT = 20;
      const auto B = 1 << SHIFT; // Big Integer (ad-hoc)
      const auto prB = this->perturb_rate * B;
      if (ub == K2 && nbd.back().similarity == s && (rand() & (B - 1)) >= prB)
      { // ub == K2 && rand() <= perturb_rate
        return NOT_INSERTED;
      }

      // search lb
      const std::size_t lb = std::lower_bound(nbd.begin(), nbd.begin() + ub, joiner_elem, comp_similarity()) - nbd.begin();

      if (K2 > 0 && nbd[lb].similarity == s)
      {
        for (std::size_t i = lb; i < ub; ++i)
        {
          if (nbd[i].node_id == joiner)
          {
            return NOT_INSERTED;
          } // joiner still in nbd
        }
      }

      assert(lb <= ub);
      auto pos = (lb < ub) ? lb + rand() % (ub - lb) : lb;

      // insert
      if (K2 < this->K)
      {
        nbd.insert(nbd.begin() + pos, joiner_elem);
        chkd.insert(chkd.begin() + pos, false); //与原论文相反，原论文为true
      }
      else
      {
        assert(K2 == this->K);
        nbd_elem_type cur_elem(joiner_elem);
        bool cur_checked = false; //与原论文相反，原论文为true
        //对应UpdateNN，由于用了一个有序vector，只要保证vector有序即等于UpdateNN。
        for (std::size_t i = pos; i < K2; ++i)
        {
          std::swap(cur_elem, nbd[i]);
          std::swap(cur_checked, chkd[i]);
        }
      }

      return INSERTED;
    }

    // 对应原论文生成old和new邻居
    void prep_nbd(const std::size_t i, nbd_type &new_nbd, nbd_type &old_nbd)
    {
      const std::size_t N = this->nodes.size();
      const nbd_type &nbd(this->nbds[i]);
      const std::size_t K2 = nbd.size();
      const std::size_t rhoK = std::min(static_cast<std::size_t>(std::ceil(this->rho * this->K)), N - 1);

      std::vector<std::size_t> sampled;
      for (std::size_t j = 0, n = 0; j < K2; ++j)
      {
        assert(nbd[j].node_id != i);
        if (this->checked[i][j])
        {
          // 和原论文是反的，原论文是false，才是old。
          old_nbd.push_back(nbd[j]);
        }
        else
        {
          // choose rhoK unchecked element with reservoir sampling
          if (n < rhoK)
          {
            sampled.push_back(j);
          }
          else
          {
            std::size_t m = rand() % (n + 1);
            if (m < rhoK)
            {
              sampled[m] = j;
            }
          }
          ++n;
        }
      }

      for (const std::size_t j : sampled)
      {
        assert(i != nbd[j].node_id);
        this->checked[i][j] = true; //和原论文是反的，原论文是mark sampled items as false
        new_nbd.push_back(nbd[j]);
      }
    }

    // 论文里的局部连接
    std::size_t local_join(const std::size_t i, nbd_type &new_nbd, nbd_type &old_nbd,
                           const nbd_type &new_rnbd, const nbd_type &old_rnbd)
    {

      std::size_t update_count = 0;
      const std::size_t N = this->nodes.size();
      const std::size_t rhoK = std::min(static_cast<std::size_t>(std::floor(this->rho * this->K)), N - 1);

      // old_nbd = old_nbd \cup sample(old_rnbd, rhoK)
      for (const auto &elem : old_rnbd)
      {
        if (rand() % old_rnbd.size() < rhoK)
        {
          old_nbd.push_back(elem);
        }
      }

      // new_nbd = new_nbd \cup sample(new_rnbd, rhoK)
      for (const auto &elem : new_rnbd)
      {
        if (rand() % new_rnbd.size() < rhoK)
        {
          new_nbd.push_back(elem);
        }
      }

      // join(new_nbd, this)
      for (const auto &elem : new_nbd)
      {
        assert(elem.node_id != i);
        update_count += this->join(elem.node_id, i);
      }

      // join(new_nbd, new_ndb)
      for (std::size_t j1 = 0, M = new_nbd.size(); j1 < M; ++j1)
      {
        const auto &elem1(new_nbd[j1]);
        for (std::size_t j2 = j1 + 1; j2 < M; ++j2)
        {
          const auto &elem2(new_nbd[j2]);
          if (elem1.node_id == elem2.node_id)
          {
            continue;
          }
          update_count += this->join(elem1.node_id, elem2.node_id);
          update_count += this->join(elem2.node_id, elem1.node_id);
        }
      }

      // join(new_nbd, old_ndb)
      for (const auto &elem1 : new_nbd)
      {
        for (const auto &elem2 : old_nbd)
        {
          if (elem1.node_id == elem2.node_id)
          {
            continue;
          }
          update_count += this->join(elem1.node_id, elem2.node_id);
          update_count += this->join(elem2.node_id, elem1.node_id);
        }
      }

      // random_join
      for (std::size_t j = 0; j < this->num_random_join; ++j)
      {
        std::size_t node_id = rand() % (this->nodes.size() - 1);
        if (node_id >= i)
        {
          ++node_id;
        }
        this->join(i, node_id);
      }

      return update_count;
    }

  public:
    uint32_t K;
    uint32_t num_random_join; // random join size
    double rho;               // sample rate
    double perturb_rate;
    std::vector<node_type> nodes;
    std::vector<nbd_type> nbds;
    rand_type rand;
    sim_type sim;

  private:
    std::vector<std::vector<bool>> checked;

  }; // end of struct NNDescent

}; // namespace nng
