import asaclient
import json
import time
import sys
from os.path import exists


def prepare_comp(comp: dict) -> dict:
    comp['alias'] = {}
    for name in comp['subcomponents']:
        for comp, i in zip(comp['subcomponents'][name], range(len(comp['subcomponents'][name]))):
            comp['subcomponents'][name][i] = prepare_comp(comp['subcomponents'][name][i])
    return comp

def prepare_sim(sim: asaclient.Simulation) -> asaclient.Simulation:
    tac_view_rec = {
        "identifier": "AsaWsTacviewRecorder@AsaModels",
        "attributes": {},
        "subcomponents": {}
    }
    sim = dict(sim)
    sim['station']['subcomponents']['recorders'] = [tac_view_rec,]
    return sim

    
    sim['station'] = prepare_comp(sim['station'])
    
    return asaclient.Simulation(**sim)

def create_sim(file) -> asaclient.Simulation:
    with open(file, "r") as f:
        sim = json.load(f)

    ssim = asaclient.Simulation(**sim)

    return asa.save_simulation(ssim)

asa = asaclient.ASA(server="http://service.homolog.asa.dcta.mil.br")

if len(sys.argv) < 2:
    print("Faltando nome do arquivo da simulacao (simulation.json). ")
    exit(1)
    
file = sys.argv[1]
    
if not exists(file):
    print(f"Arquivo {file} nao encontrado")
    exit(1)

sim = create_sim(file)

if not sim:
    print("Erro criando a simulação")
    exit(1)

print(f"Simulação {sim.name} #{sim.id} criada.")

body = prepare_sim(sim=sim)

# exit(0)

result = asa.session._post("execution", json=body)

if result.status_code == 200:
    exec_uuid = json.loads(result.content)['uuid']
    print(f"Execução criada {exec_uuid}")
else:
    print("Erro ao iniciar execução: ")
    print(result.content)
    exit(1)


time.sleep(10)

result = asa.session._put(f"execution/{exec_uuid}/start", json={'speed': 1000})

if result.status_code != 200:
    print("Erro ao iniciar simulacao")
    print(result.content)
    exit(1)


print("Aguardando execução: ")
last_status = None
while True:
    
    exec = asa.execution(exec_uuid)
    if exec.status != last_status:
        print(f"Status da execução: {exec.status.name}.")
        last_status = exec.status
    if exec.ended():
        if exec.finished():
            result = asa.session._get(f"execution/{exec_uuid}/tacview")
            if result.status_code == 200:
                file = f"{exec_uuid}.acmi"
                with open(file, "wb") as f:
                    f.write(result.content)
                    f.flush()
                    print()
                    print(f"Arquivo salvo {file}")
                    exit(0)
            else:  
                print(f"Ocorreu algum erro durante a execução: {exec}")
                print(result.content)
                exit(1)
        else:
            print(f"A simulação encerrou sem sucesso: {exec}")
            exit(1)
    else:
        time.sleep(1)
    