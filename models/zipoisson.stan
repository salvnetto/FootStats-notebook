data {
  //train
  int<lower=1> nteams;
  int<lower=1> ngames;
  array[ngames] int team1;
  array[ngames] int team2;
  array[ngames] int<lower=0> gf;
  array[ngames] int<lower=0> ga;
  //test
  int<lower=1> ngames_new;
  array[ngames_new] int team1_new;
  array[ngames_new] int team2_new;
}

parameters {
  sum_to_zero_vector[nteams] att;
  sum_to_zero_vector[nteams] def;
  real home;
  real<lower=0, upper=1> pzero_x;
  real<lower=0, upper=1> pzero_y;
}

transformed parameters {
  vector[ngames] theta1;
  vector[ngames] theta2;

  theta1 = (att[team1] - def[team2] + home);
  theta2 = (att[team2] - def[team1]);
}

model {
  // Priors
  att ~ normal (0, 10);
  def ~ normal (0, 10);
  home ~ normal (0, 10);
  pzero_x ~ beta(1, 1);
  pzero_y ~ beta(1, 1);

  // Likelihood
  for (k in 1:ngames){
    // HOME
    if (gf[k] == 0)
      target += log_sum_exp(
        bernoulli_lpmf(1 | pzero_x),
        bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(gf[k] | theta1[k])
        );
    else
      target += bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(gf[k] | theta1[k]);

    // AWAY
    if (ga[k] == 0)
      target += log_sum_exp(
        bernoulli_lpmf(1 | pzero_y),
        bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(ga[k] | theta2[k])
        );
    else
      target += bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(ga[k] | theta2[k]);
  }
}

generated quantities {
  vector[ngames] log_lik;         // Log-likelihood for model comparison
  vector[ngames_new] theta1_new;  // Expected goals for new home teams
  vector[ngames_new] theta2_new;  // Expected goals for new away teams
  array[ngames_new] int gf_new;
  array[ngames_new] int ga_new;

  // Log-likelihood for training data
  for (k in 1:ngames) {
    log_lik[k] = 0;
    // HOME
    if (gf[k] == 0)
      log_lik[k] += log_sum_exp(
        bernoulli_lpmf(1 | pzero_x),
        bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(gf[k] | theta1[k])
        );
    else
      log_lik[k] += bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(gf[k] | theta1[k]);

    // AWAY
    if (ga[k] == 0)
      log_lik[k] += log_sum_exp(
        bernoulli_lpmf(1 | pzero_y),
        bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(ga[k] | theta2[k])
        );
    else
      log_lik[k] += bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(ga[k] | theta2[k]);
  }

  // Predictive distributions for test data
  theta1_new = (att[team1_new] - def[team2_new] + home);
  theta2_new = (att[team2_new] - def[team1_new]);

  for (k in 1:ngames_new) {
    gf_new[k] = poisson_log_rng(theta1_new[k]);
    if (gf_new[k] == 0)
      gf_new[k] = bernoulli_rng(pzero_x) == 1 ? 0 : poisson_log_rng(theta1_new[k]);

    // AWAY PREDICTIONS
    ga_new[k] = poisson_log_rng(theta2_new[k]);
    if (ga_new[k] == 0)
      ga_new[k] = bernoulli_rng(pzero_y) == 1 ? 0 : poisson_log_rng(theta2_new[k]);

  }
}

