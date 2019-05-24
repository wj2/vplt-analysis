
data {
//sizes 
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of outcomes
  int<lower=0> L; // number of images
  
  // prior data
  real<lower=0> prior_eps_var;
  real<lower=0> prior_bias_var;
  real<lower=0> prior_salience_var;
  real<lower=0> prior_tnov_mean;
  real<lower=0> prior_tfam_mean;
  real<lower=0> prior_tnov_var;
  real<lower=0> prior_tfam_var;

  // main data
  int imgs[N, L, K] ; // image selection matrix
  matrix[N, K] novs; // novel indicator
  matrix[N, K] views; // number of previous views
  int y[N]; // outcome vector
}

parameters {
  vector[L] s; // image inherent saliences
  vector<lower=0>[K] bias; // target bias terms
  real eps; // novelty bias
  real<lower=0.0001, upper=10000> tau_nov; // familiarity decay
  real<lower=0.0001, upper=10000> tau_fam; // familiarity decay
}

model {
  // var declarations
  matrix[N, K] img_s;
  matrix[N, K] outcome_evidence;
  matrix[N, K] tau_matrix;

  // priors
  eps ~ normal(0, prior_eps_var);
  bias ~ normal(0, prior_bias_var);
  s ~ normal(0, prior_salience_var);
  tau_nov ~ normal(prior_tnov_mean, prior_tnov_var);
  tau_fam ~ normal(prior_tfam_mean, prior_tfam_var);
  
  // setup
  tau_matrix = novs*tau_nov + (1 - novs)*tau_fam;
  
  for (k in 1:K) {
    img_s[:, k] = to_matrix(imgs[:, :, k])*s;
  }
  
  outcome_evidence = (img_s + novs * eps) .* exp(-views ./ tau_matrix);
  
  for(k in 1:K) {
    outcome_evidence[:, k] = outcome_evidence[:, k] + bias[k];
  }
  
  for (n in 1:N) {
    y[n] ~ categorical_logit(outcome_evidence[n]');
  }
}
