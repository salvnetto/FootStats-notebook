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
  real<lower=0> phi_home;
  real<lower=0> phi_away;
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
  phi_home ~ cauchy(0, 25);
  phi_away ~ cauchy(0, 25);

  // Likelihood
  for (k in 1:ngames) {
    target += neg_binomial_2_log_lpmf(gf[k] | theta1[k], phi_home);
    target += neg_binomial_2_log_lpmf(ga[k] | theta2[k], phi_away);
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
    log_lik[k] = neg_binomial_2_log_lpmf(gf[k] | theta1[k], phi_home) +
                 neg_binomial_2_log_lpmf(ga[k] | theta2[k], phi_away);
  }

  // Predictive distributions for test data
  theta1_new = (att[team1_new] - def[team2_new] + home);
  theta2_new = (att[team2_new] - def[team1_new]);

  gf_new = neg_binomial_2_log_rng(theta1_new, phi_home);
  ga_new = neg_binomial_2_log_rng(theta2_new, phi_away);
}
