data {
  //train
  int<lower=1> nteams;
  int<lower=1> ngames;
  array[ngames] int<lower=1, upper=nteams> team1;
  array[ngames] int<lower=1, upper=nteams> team2;
  array[ngames] int<lower=0> gf;
  array[ngames] int<lower=0> ga;
  //test
  int<lower=1> ngames_new;
  array[ngames_new] int<lower=1, upper=nteams> team1_new;
  array[ngames_new] int<lower=1, upper=nteams> team2_new;
}

parameters {
  sum_to_zero_vector[nteams] att;
  sum_to_zero_vector[nteams] def;
  sum_to_zero_vector[nteams] home;
}

transformed parameters {
  vector[ngames] theta1;
  vector[ngames] theta2;

  theta1 = (att[team1] - def[team2] + home[team1]);
  theta2 = (att[team2] - def[team1]);
}

model {
  // Priors
  att ~ normal (0, 10);
  def ~ normal (0, 10);
  home ~ normal (0, 10);

  // Likelihood
  for (k in 1:ngames) {
    target += poisson_log_lpmf(gf[k] | theta1[k]);
    target += poisson_log_lpmf(ga[k] | theta2[k]);
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
    log_lik[k] = poisson_log_lpmf(gf[k] | theta1[k]) +
                 poisson_log_lpmf(ga[k] | theta2[k]);
  }

  // Predictive distributions for test data
  theta1_new = (att[team1_new] - def[team2_new] + home[team1_new]);
  theta2_new = (att[team2_new] - def[team1_new]);

  gf_new = poisson_log_rng(theta1_new);
  ga_new = poisson_log_rng(theta2_new);
}
