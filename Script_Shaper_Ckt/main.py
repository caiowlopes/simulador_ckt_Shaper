import numpy as np
import Gerador_de_Sinais as GS
import Funcoes_auxiliares as FA
import Shaper_Simulator

# Constantes #

qntd_canais = 10  # qntd de listas
amostras_por_leitura = 10  # tamanho da lista
qntd_leitura_por_canal = 10  # usado para as leituras e medias
seed = 42


# Ondas/resp aos pulsos gerados
ckt_parameters_error = (  # dois ultimos são tau1 e tau2
    np.array([10, 1, 1, 1, 2, 2, 2, 0.10, 0, 0], dtype=float)
) / 100

# cada linha é um canal
waves_por_canal = np.asarray(  # waves_por_canal[0] = sinal sem variação dos componentes
    GS.wave_former_main(
        num_canais=qntd_canais,
        qntd_amostras=amostras_por_leitura,
        noise=ckt_parameters_error,
        CKT_parameters=Shaper_Simulator.ckt_parameters,
        CKT_simulator=Shaper_Simulator.MonteCarlo_iteration,
        Seed=seed,
        retornar_t_resp=False,
    )
)

# Sinais Originais #
s_originais = {}  # s_originais
count = 0  # para modificar os sinais entre si

for canais in range(qntd_canais):
    s_originais[f"canal_{canais}"] = []

    for amostra in range(qntd_leitura_por_canal):
        s_originais[f"canal_{canais}"].append(
            GS.original_signal_generator(amostras_por_leitura, seed=seed + count)
        )
        count += 3

# Readouts #
leituras = {}
for (canal, amostras), wave in zip(s_originais.items(), waves_por_canal):
    leituras[canal] = [np.convolve(amostra, wave, mode="full") for amostra in amostras]
print("ok")
