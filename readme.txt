This code repository includes two primary code files:

Note that to run both the files, installation of the OpenAI Gym toolkit is needed. The instructions to install Open AI
Gym can be found at - https://gym.openai.com/docs/

1). two_time_ts_mc.py - for the Mountain Car problem.

2). two_time_ts_pl.py - for the Inverted Pendulum problem.

The files implement GTD2 and TDC algorithms on their respective domains. In both the files:

1). The variable step_rule takes two possible values: "gupta" and "mannor". The former value implements the proposed
adaptive learning rate rule and the latter implements the optimal polynomial decay rule suggested in literature.

2). The variables step_ini, update_threshold, update_factor and replay refer to the hyperparameters rho, sigma, xi and
N in the adaptive learning rate rule as described in Algorithm 1 in the paper.

3). The variables step_num_mannor, alpha and beta refer to the hyperparameters rho_0, alpha and beta in the optimal
polynomial decay rule.

4). err_arr stores the errors. It is a num_episodes x num_iter array and stores the error across different runs to
allow computation of standard deviation and mean of errors.



