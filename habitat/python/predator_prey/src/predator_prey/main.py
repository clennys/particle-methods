from predator_prey.sim import Simulation


if __name__ == "__main__":
    domain_size = 10
    initial_rabbits = 900
    initial_wolves = 100
    rabbit_max_age = 100
    wolf_hunger_threshold = 50
    step_std = 0.5
    capture_radius = 0.5
    eat_prob = 0.02
    rabbit_repl_prob = 0.02
    n_subgrids = 20
    
    sim = Simulation(
        domain_size=domain_size,
        initial_rabbits=initial_rabbits,
        initial_wolves=initial_wolves,
        rabbit_max_age=rabbit_max_age,
        wolf_hunger_threshold=wolf_hunger_threshold,
        step_std=step_std,
        capture_radius=capture_radius,
        eat_prob=eat_prob,
        rabbit_repl_prob=rabbit_repl_prob,
        n_subgrids=n_subgrids
    )

    sim.run(time_steps=4000)
    sim.plot_population_dynamics("ex01")

    rabbit_max_age = 50
    
    sim1 = Simulation(
        domain_size=domain_size,
        initial_rabbits=initial_rabbits,
        initial_wolves=initial_wolves,
        rabbit_max_age=rabbit_max_age,
        wolf_hunger_threshold=wolf_hunger_threshold,
        step_std=step_std,
        capture_radius=capture_radius,
        eat_prob=eat_prob,
        rabbit_repl_prob=rabbit_repl_prob,
        n_subgrids=n_subgrids
    )

    sim1.run(time_steps=4000)
    sim1.plot_population_dynamics("ex02")

    rabbit_max_age = 100
    step_std = 0.05
    domain_size = 8

    sim2 = Simulation(
        domain_size=domain_size,
        initial_rabbits=initial_rabbits,
        initial_wolves=initial_wolves,
        rabbit_max_age=rabbit_max_age,
        wolf_hunger_threshold=wolf_hunger_threshold,
        step_std=step_std,
        capture_radius=capture_radius,
        eat_prob=eat_prob,
        rabbit_repl_prob=rabbit_repl_prob,
        n_subgrids=n_subgrids
    )

    sim2.run(time_steps=4000)
    sim2.plot_population_dynamics("ex03")
    

    
    # Optional: Run with visualization
    # new_sim = Simulation(
    #     domain_size=domain_size,
    #     initial_rabbits=initial_rabbits,
    #     initial_wolves=initial_wolves,
    #     rabbit_max_age=rabbit_max_age,
    #     wolf_hunger_threshold=wolf_hunger_threshold,
    #     step_std=step_std,
    #     capture_radius=capture_radius,
    #     eat_prob=eat_prob,
    #     rabbit_repl_prob=rabbit_repl_prob,
    #     n_subgrids=n_subgrids
    # )
    # new_sim.run(time_steps=500, plot_interval=5)
