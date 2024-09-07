#ifndef STAN_MATH_PRIM_PROB_CU_BETA_NEG_BINOMIAL_LPMF_HPP
#define STAN_MATH_PRIM_PROB_CU_BETA_NEG_BINOMIAL_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/lbeta.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/operands_and_partials.hpp>
#include <cmath>
#include <stan/math.hpp>


extern "C" {
  void calculateLpmf(const int* n_data, const double* r_data, const double* a_data, const double* b_data, double* logp, double* p1, double* p2, double* p3, int vec_size);
}


namespace model_cu_bnb_model_namespace {

/** \ingroup prob_dists
 * Returns the log PMF of the Beta-Negative Binomial distribution with given 
 * number of successes, prior success, and prior failure parameters. 
 * Given containers of matching sizes, returns the log sum of probabilities.
 *
 * @tparam T_n type of failure parameter
 * @tparam T_r type of number of successes parameter
 * @tparam T_size1 type of prior success parameter
 * @tparam T_size2 type of prior failure parameter
 * 
 * @param n failure parameter
 * @param r Number of successes parameter
 * @param alpha prior success parameter
 * @param beta prior failure parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if r, alpha, or beta fails to be positive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <bool propto, typename T_n, typename T_r, typename T_size1,
          typename T_size2,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_n, T_r, T_size1, T_size2>* = nullptr>
inline stan::return_type_t<T_r, T_size1, T_size2> cu_beta_neg_binomial_lpmf(const T_n& n, 
                                                    const T_r& r,
                                                    const T_size1& alpha,
                                                    const T_size2& beta,
                                                    std::ostream* pstream__) {

  using stan::partials_return_t;
  using stan::ref_type_t;
  
  using stan::is_constant;
  using stan::is_constant_all;
  using stan::VectorBuilder;
  using stan::scalar_seq_view;
  using stan::math::lgamma;
  using stan::math::size;
  using stan::math::max_size;
  using namespace stan::math;

  using T_partials_return = partials_return_t<T_n, T_r, T_size1, T_size2>;
  using T_r_ref = ref_type_t<T_r>;
  using T_alpha_ref = ref_type_t<T_size1>;
  using T_beta_ref = ref_type_t<T_size2>;
  static const char* function = "cu_beta_neg_binomial_lpmf";
  check_consistent_sizes(function, "Successes variable", n,
                         "Number of successes parameter", r,
                         "First prior sample size parameter", alpha,
                         "Second prior sample size parameter", beta);
  if (size_zero(n, r, alpha, beta)) {
    return 0.0;
  }

  T_r_ref r_ref = r;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;
  check_positive_finite(function, "Number of successes parameter", r_ref);
  check_positive_finite(function, "First prior sample size parameter", alpha_ref);
  check_positive_finite(function, "Second prior sample size parameter", beta_ref);

  if (!include_summand<propto, T_r, T_size1, T_size2>::value) {
    return 0.0;
  }


  T_partials_return logp(0.0);
  operands_and_partials<T_r_ref, T_alpha_ref, T_beta_ref> ops_partials(r_ref, alpha_ref, beta_ref);

  scalar_seq_view<T_n> n_vec(n);
  scalar_seq_view<T_r_ref> r_vec(r_ref);
  scalar_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_beta_ref> beta_vec(beta_ref);
  size_t size_n = stan::math::size(n);
  size_t max_size_seq_view = max_size(n, r, alpha, beta);


  for (size_t i = 0; i < max_size_seq_view; i++) {
    if (n_vec[i] < 0) {
      return ops_partials.build(LOG_ZERO);
    }
  }

  std::vector<int> full_n(max_size_seq_view);
  std::vector<double> full_r(max_size_seq_view);
  std::vector<double> full_alpha(max_size_seq_view);
  std::vector<double> full_beta(max_size_seq_view);
  std::vector<double> full_logp(max_size_seq_view);
  std::vector<double> p1(max_size_seq_view);
  std::vector<double> p2(max_size_seq_view);
  std::vector<double> p3(max_size_seq_view);

  for (size_t i = 0; i < max_size_seq_view; i++) {
    full_n[i] = value_of(n_vec[i]);
    full_r[i] = value_of(r_vec[i]);
    full_alpha[i] = value_of(alpha_vec[i]);
    full_beta[i] = value_of(beta_vec[i]);
  }

  calculateLpmf(full_n.data(), full_r.data(), full_alpha.data(), full_beta.data(), full_logp.data(), p1.data(), p2.data(), p3.data(), max_size_seq_view);

  
  for (size_t i = 0; i < max_size_seq_view; i++) {
    logp += full_logp[i];

    if (!is_constant_all<T_r>::value) {
      ops_partials.edge1_.partials_[i] += p1[i];
    }
    if (!is_constant_all<T_size1>::value) {
      ops_partials.edge2_.partials_[i] += p2[i];
    }
    if (!is_constant_all<T_size2>::value) {
      ops_partials.edge3_.partials_[i] += p3[i];
    }
  }
  return ops_partials.build(logp);
}

template <typename T_n, typename T_r, typename T_size1, typename T_size2>
inline stan::return_type_t<T_r, T_size1, T_size2> cu_beta_neg_binomial_lpmf(const T_n& n, 
                                                   const T_r& r,
                                                   const T_size1& alpha,
                                                   const T_size2& beta,
                                                   std::ostream* pstream__) {
  return cu_beta_neg_binomial_lpmf<false>(n, r, alpha, beta);
}








}
#endif
