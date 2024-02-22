# Bibliotecas

import matplotlib.pyplot as plt 
import sympy as sp
from sympy import Eq, Poly, exp
from sympy import fraction
from sympy.abc import s,t
import datetime
import scipy.signal as signal
import numpy as np
import random 
import warnings
warnings.simplefilter("ignore", np.ComplexWarning) 


inicio = datetime.datetime.now() 


#declaração variaveis

tau_1, tau_2, Vo, Vi   = sp.symbols('tau_1 tau_2 Vo Vi') 
C0, C1, C2, C3, C4  = sp.symbols('C0 C1 C2 C3 C4 ') 
R0, R1, R2  = sp.symbols('R0 R1 R2') 
L1, L2, L3 = sp.symbols('L1 L2 L3') 
I1, I2, I3, I4, I5, I6  = sp.symbols('I1 I2 I3 I4 I5 I6') 

# Limite do plot com 400 pontos com distancia de 25*10^-9 entre eles
t1 = np.arange(0, 400) * 25 * 10**-9 # usado para todos os termos 

# componentes do ckt 
Cord = [C0, C1, C2, C3, C4, L1, L2, L3, R0, R1, R2, tau_1, tau_2] 
Cval0 = [101e-9, 5e-12, 138e-12, 102e-12, 6.8e-12, 2.48e-6, 1.6e-6, 
            0.78e-6, 1.1e3, 75, 49.9, 3.1046e-09, 6.5798e-09] # valores exatos do Cord em ordem

# componentes do plot
y = [] # lista de todos o so graficos somados com erros
xa = [] # lista das FP a serem somadas pro grafico
y1 = [] # aux
tds_polos = [] # salvando os polos
p3 = 0 # parametro de repetiçao seno e cosseno

# lista com valor medio de todos os valores de y
ymed = [] 

# desvio padrao dos valores de y e de ymed
desv_pad = [] 

# parte real e imagianria para mapa dos polos
real_part = []
imag_part = []

# funçao Laplace
def laplace(f):
    return sp.laplace_transform(f, t, s, noconds=True) 

# equações do ckt 
eqn1 = Eq( 1/(C1*s)*(I1-I2) + 1/(C0*s)*I1 , Vi) 
eqn2 = Eq( 1/(C1*s)*(I2-I1) + s*L1*I2 + 1/(C2*s)*(I2-I3) , 0) 
eqn3 = Eq( 1/(C2*s)*(I3-I2) + s*L2*I3 + 1/(C3*s)*(I3-I4) , 0) 
eqn4 = Eq( 1/(C3*s)*(I4-I3) + s*L3*I4 + 1/(C4*s)*(I4-I5) , 0) 
eqn5 = Eq( 1/(C4*s)*(I5-I4) + (I5-I6)*R0 , 0) 
eqn6 = Eq( R0*(I6-I5) + R1*I6 + R2*I6 , 0) 
eqn7 = Eq( I6*R2, Vo ) 
eqns = [eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7] 

# soluçao do ckt 
Sol = sp.solve( eqns, (I1, I2, I3, I4, I5, I6, Vo) ) 
si6 = Sol[I6] 

# saida tirando a entrada
h = R2*si6/Vi

# PMT
v = exp(-t/tau_1) - exp(-t/tau_2) 
V = laplace(v)
display(V) 

# função trasferencia final 
H1 = -V * h
H = H1

#pocentagens de erros permitidas pra cada elemento d ckt
e = 5 # erro
# Cord = [C0, C1, C2, C3, C4, L1, L2, L3, R0, R1, R2, tau_1, tau_2] 
percents = [ e, e, e, e, e, e, e, e, e, e, e, 0, 0] # erro maximo de cada componente

# componentes do plot
y = [] # lista de todos o so graficos somados 
xa = [] # lista das FP a serem somadas pro grafico
y1 = [] # aux
tds_polos = [] # salvando os polos
p3 = 0 # parametro de repetiçao seno e cosseno

# quantidade de vezes q o loop vai rodar 
vezes = 15 # pra valer: 1500

for i in range(0, vezes):
    
    Cval0 = [101e-9, 5e-12, 138e-12, 102e-12, 6.8e-12, 2.48e-6, 1.6e-6, 
            0.78e-6, 1.1e3, 75, 49.9, 3.1046e-09, 6.5798e-09] # valores exatos do Cord em ordem
    
    # valores aleatorios
    # range d erro; erro máximo de -e% ate +e% 
    
    Cval = [] # lista d componentes com valores alterados

    for valor, erro in zip(Cval0, percents): 
        
        if erro == 0:
            variacao = erro 
        else:
            variacao = (random.randrange(-erro*100, erro*100)) /10000
                    
        aux = valor * variacao + valor
        #print(variacao) 
        # lista com valores alterados
        Cval.append(aux) # valores de Cval com erro/variação de +-e% 

    if i == 0: # valor do sinal real/sem erros
        Cval = Cval0
        
    # função trasferencia 
    H = H1 
    
    # Substituição de valores
    for variavel, valor1 in zip(Cord, Cval): 
        H = H.subs(variavel, valor1)

    # Separando numerador dedenominador 
    N_H, D_H = fraction(H) 

    
    '''RESIDUOS E POLOS'''

    # Coeficientes do numerador e denominador
    
    coefs_num = [] # 'zerando' variavel
    coefs_den = [] # 'zerando' variavel
    
    coefs_num = sp.Poly(N_H, s).all_coeffs() 
    coefs_den = sp.Poly(D_H, s).all_coeffs()

    # frações parciais
    residuo, polo, b0 = [], [], [] # 'zerando' variavel
    
    residuo, polo, b0 = signal.residue(coefs_num, coefs_den) 
    
    # salvando todos os polos
    tds_polos.append(polo) 
    
    # Correção do residuo (tirando a parte img dos residuos reais) 
    for j in range(0, len(polo)):
        if polo[j].imag == 0: 
            residuo[j] = residuo[j].real 

    '''LAPLACE INVERSA E GRAFICOS'''
    
    for k in range (0, len(polo)): 
        if polo[k].imag == 0: 
            residuo[k] = residuo[k].real 
            polo[k] = polo[k].real

        # Verifique se a parte imaginária esta zerada
        if polo[k].imag == 0:
            
            'EXPONENCIAIS'
            
            A = residuo[k] # ganho
            d = polo[k] # taxa d decaimento 
            x = A * np.exp(d * t1)
            
            xa.append(x)
            
        else:
            'Senos e cossenos'
            
            pol = polo[k]
            pol_1 = polo[k-1] #  auxiliar
            
            if pol != pol_1 and pol != np.conjugate(pol_1):
                
                resi = residuo[k] 
                
                a1 = pol.real # parte real polo
                b1 = abs(pol.imag) # parte imaginaria polo

                Modulo = abs(resi) # modulo residuo
                fase = (np.angle(resi)) # fase residuo em rad

                # termo FP
                x = 2*Modulo*np.exp(a1*t1) * np.cos( b1*t1 + fase)

                xa.append(x) 
                
    'SOMA'
    if i == 0: # sinal sem variaçoes/sem erros
        sinal0 = sum(xa)
        sinal = sinal0 / (max(abs(sinal0))) # pegar maior modulo/ normalizando
        y.append(sinal)
    else: 
        y1 = sum(xa) # soma / FPs somados; salvar y1 em excel 

        # Normalizando 
        y2 = y1 / (max(abs(y1))) # pegar maior modulo

        y.append(y2)  

# media da lista
ymed = [] 
ymed = [sum(item) / len(item) for item in zip(*y)] # nao normalizar

# Calculando o desvio padrão 
desv_pad = []
desv_pad = np.std(y, axis=0) 

# Plot Sinal sem erro normalizado

# Plotar o gráfico
plt.plot(t1, y[1], color= 'b', linewidth = 2) #, label='y = x^2') linewidth = expessura
plt.xlabel('t') # titulo eixo x
plt.ylabel('f(t)') # titulo eixo y 
plt.title('Frações Parciais Somadas') # titulo Grafico
plt.grid(True) # fundo do grafico com grade
plt.xlim(-0.25*10**-6, 0.5*10**-6)  # Limites do eixo x
plt.axhline(0, color='black', linewidth=0.65)
plt.axvline(0, color='black', linewidth=0.65)
plt.show() 


# Plotagem

banda_sup = ymed + desv_pad*2.5
banda_inf = ymed - desv_pad*2.5

plt.figure(figsize=(12, 6)) 

# Sinais 
plt.plot( sinal, label='Pulso sem varição',linestyle='-',color = 'b' ,linewidth=2)
plt.plot( ymed, label='Média erros', linestyle='--', color = 'orange', linewidth=2)
plt.plot( banda_sup, label='Banda Superior', linestyle='-.',color = 'g', linewidth=2)
plt.plot( banda_inf, label='Banda Inferior', linestyle='-.',color = 'r', linewidth=2) 

# Sombrear a área entre a banda_sup e a banda_inf
plt.fill_between(range(len(banda_sup)), banda_sup, banda_inf, color='gray', alpha=0.5)

# Eixos
w = 1.3
plt.axhline(0, color='black', linewidth=w)
plt.axvline(0, color='black', linewidth=w) 

plt.title("Bandas de Bollinger")  
plt.xlim(-0.1, 30) 
plt.legend() 
plt.grid(True)
plt.show()  

# mapa de polos

real_part = []
imag_part = []

for i in tds_polos:
    # Separar parte real e parte imaginária
    real_part = np.real(i)
    imag_part = np.imag(i) 
    
    # gráfico
    plt.scatter(real_part, imag_part, marker='x')
    
# eixos
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# titulo 
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginária')
plt.title('Plano Complexo')

# Grade
plt.grid(True, linestyle='-', alpha=0.7)

plt.show() # mapa de polos