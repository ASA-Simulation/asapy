import asapy
import asaclient
import json
import pandas as pd
import os
asa = asaclient.ASA(server="https://service.asa.dcta.mil.br")

sim = asapy.load_simulation(f"./simulations/nav_demo.json")  # Creating a Simulation from the JSON file at the specified path, and assigning it to 'sim'.
sim = asa.save_simulation(sim)

doe = asapy.Doe()
aliases = doe.process_aliases_by_sim(sim, asa.component_configs()) 
df = doe.create(aliases, samples=9, seed=42)

metric = 'acft_standing' 
side = 'blue'

batch = asaclient.Batch(label=simulation_name, simulation_id=sim.id)
batch = asa.save_batch(batch)
print(f"Batch criado: {batch.id}")
ec = asapy.ExecutionController(sim_func=asapy.batch_simulate(batch=batch), stop_func=asapy.stop_func(metric=metric, threshold=0.001, side=side), chunk_size=3)

result = ec.run(doe=df)