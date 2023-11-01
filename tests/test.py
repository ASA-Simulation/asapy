import asapy
import time

asa = asapy.Asa()

sims = asa.simulations()
sim = asa.simulation(1)

configs = asa.component_configs()
doe = asapy.Doe()
aliases = doe.process_aliases_by_sim(sim, configs)
df = doe.create(aliases, samples=10)
metrics = doe.process_metrics(sim)

exps = doe.prepare_experiments(df)

# criar batch
batch = asapy.Batch(label="batch_001", simulation_id=sim.id)
# enviar batch para o manager
batch = asa.save_batch(batch)
# roda batch
#execs = batch.add_chunks(exps)
executions = batch.run(exps, metric = 'msl_remaining')


while True:
    my_batch = asa.batch(1)

    s = my_batch.status()
    print(s)
    time.sleep(2)


# Verificar batches de uma sim id
# batches = sim.batches()
# my_batch = asa.batch(1)
# metrics = my_batch.records(types=["asa::recorder::AsaMonitorReport"])