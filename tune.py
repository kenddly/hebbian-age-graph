import optuna
import numpy as np

# Import your existing environment and graph
from env import SnakeEnv, MAX_STEPS
from graph import AgeingBipartiteGraph

# ─── TRAINING & EVALUATION UTILITIES ────────────────────────────────────────

def run_episode(agent, env, train=True):
    state = env.reset()
    total_r, food = 0.0, 0

    for _ in range(MAX_STEPS):
        action = agent.forward(state) if train else agent.predict(state)
        state, reward, done = env.step(action)

        if train:
            agent.apply_reward(reward)

        total_r += reward
        if reward >= 10:
            food += 1

        if done:
            break

    if train:
        agent.reset_traces()

    return total_r, env.steps, food

def evaluate(agent, n_eval_eps=10):
    env = SnakeEnv()
    rewards = []

    for _ in range(n_eval_eps):
        r, _, _ = run_episode(agent, env, train=False)
        rewards.append(r)

    return np.mean(rewards)

# ─── OPTUNA OBJECTIVE FUNCTION ─────────────────────────────────────────────

def objective(trial):
    # 1. Define the hyperparameter search space
    # Added age tuning (searching continuously from "Young" to "Very Old")
    age = trial.suggest_float("age", 0.0, 50.0) 
    
    # Using log scales for learning rates helps explore small magnitudes effectively
    trace_decay = trial.suggest_float("trace_decay", 0.7, 0.99)
    base_lr = trial.suggest_float("base_lr", 1e-4, 5e-2, log=True)
    crystallization_threshold = trial.suggest_float("crystallization_threshold", 0.5, 5.0)
    rigidity = trial.suggest_float("rigidity", 0.1, 1.0)
    baseline_lr = trial.suggest_float("baseline_lr", 1e-3, 1e-1, log=True)

    # 2. Instantiate Environment and Agent
    env = SnakeEnv()
    
    # Pass the actively tuned age to the agent
    agent = AgeingBipartiteGraph(
        num_inputs=8, 
        num_outputs=3, 
        age=age,
        trace_decay=trace_decay,
        base_lr=base_lr,
        crystallization_threshold=crystallization_threshold,
        rigidity=rigidity,
        baseline_lr=baseline_lr
    )

    # Shorter training runs for tuning (saves time, scaling up for final models)
    n_episodes = 800  
    eval_every = 100
    best_eval_reward = -np.inf

    # 3. Training loop
    for ep in range(1, n_episodes + 1):
        run_episode(agent, env, train=True)

        # Evaluate periodically
        if ep % eval_every == 0:
            eval_reward = evaluate(agent, n_eval_eps=15)

            # Report intermediate objective value to Optuna
            trial.report(eval_reward, ep)

            # If the run is performing horribly, Optuna prunes (aborts) it early
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward

    # The metric Optuna will try to maximize
    return best_eval_reward

# ─── MAIN TUNING EXECUTION ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a study object. 
    # The MedianPruner cuts off trials that perform worse than the median of previous trials
    study = optuna.create_study(
        direction="maximize", 
        study_name="snake_hebbian_tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200)
    )
    
    # Run 50 trials. Increase n_trials if you want a more exhaustive search.
    print("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=200)

    print("\n" + "="*40)
    print("Tuning Complete!")
    print("="*40)
    
    trial = study.best_trial
    print(f"Best Evaluation Reward: {trial.value:.2f}")
    print("Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
