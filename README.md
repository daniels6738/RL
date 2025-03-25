Feito para a disciplina de Aprendizagem por Reforço
Implementação de n-step TD-Learning off-policy para ambientes Gymnasium

<h1>Configurações Iniciais</h1>

<h1>1 - TD Learning Off-Policy de n Passos</h1>
  O TD Learning Off-Policy de n passos é uma extensão do TD Learning que utiliza uma sequência de n passos para atualizar a estimativa do  Q(s,a) .
  
<h2>Relembrando o Off-Policy</h2>
  No aprendizado off-policy, a política de comportamento (que escolhe as ações) é diferente da política alvo (que é avaliada). Um exemplo clássico é o Q-Learning.
  
  <h2>TD Learning On-Policy</h2>
    No TD Learning on-policy, a mesma política é usada tanto para escolher as ações quanto para avaliar a política. Um exemplo é o SARSA.
    
  <h2>Correção no TD Learning Off-Policy</h2>

No TD Learning off-policy, a estimativa do valor de uma política é atualizada a partir de trajetórias coletadas por outra política. Para corrigir essa discrepância, devem ser utilizadas técnicas como o Ordinary Importance Sampling e o Weighted Importance Sampling.

A chave para essa correção é o fator de importância (rho, ρ), definido como a razão entre a probabilidade de uma sequência de ações sob a política alvo (π) e sob a política de comportamento (b):


$ρ_t = \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$


Quando temos uma trajetória de n passos, o peso acumulado é o produto dos fatores de importância ao longo dos passos:


$ρ = \prod_{t=1}^{n} \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$


Esse fator é fundamental porque ajusta a contribuição das amostras geradas pela política de comportamento para refletirem corretamente a política alvo. Quando a política alvo e a de comportamento são muito diferentes, os valores de ρ podem variar drasticamente, tornando o aprendizado instável. O Weighted Importance Sampling resolve esse problema normalizando os pesos acumulados.

<h2>Exemplos Simples</h2>

Abaixo estão exemplos de como essa correção é calculada para diferentes valores de n.

<h3>Ordinary Importance Sampling (n=1)</h3>

Para um episódio de um passo:
- Experiência: $( s, a, r_1, s_1, a_1 )$
- Estimativa: $Q_{target} = r_1 + \gamma Q(s_1, a_1) \cdot \frac{\pi(a_1|s_1)}{b(a_1|s_1)}$


<h3>Weighted Importance Sampling (n=2)</h3>

Para um episódio de dois passos:
- Experiência: $( s, a, r_1, s_1, a_1, r_2, s_2, a_2 )$
- Estimativa:
$Q_{target} = r_1 + \gamma r_2 + \gamma^2 Q(s_2, a_2) \cdot \frac{\pi(a_1|s_1) \pi(a_2|s_2)}{b(a_1|s_1) b(a_2|s_2)}$


O Weighted Importance Sampling normaliza esse peso para reduzir a variabilidade excessiva e melhorar a estabilidade do aprendizado.

<h2>Resumo</h2>
- On-Policy (ex: SARSA): A política que escolhe as ações é a mesma que está sendo avaliada.<br>
- Off-Policy (ex: Q-Learning, TD Off-Policy): A política que coleta experiências é diferente da política alvo.<br>
- Correção via Importance Sampling: Ajusta as atualizações de Q para refletirem a política alvo corretamente.<br>
- Rho (ρ) é o fator de ajuste: Ele compensa as diferenças entre as probabilidades das políticas de comportamento e alvo.<br><br>

Esses conceitos garantem que o aprendizado ocorra de forma eficaz, mesmo quando a política de comportamento é diferente da política alvo.




  <h2>O que foi implementado?</h2>
    Nesse projeto fizemos uso de Ordinary Importance Sampling para o aprendizado de ambientes discretos e contínuos em ambientes do Gymnasium.

  <h1>2 - Sobre os ambientes</h1>
    Escolhemos os ambientes discretos CliffWalking-v0, FrozenLake-v1 e Blackjack-v1; e escolhemos o ambiente contínuo Acrobot-v1.

  <h2>2.1 - CliffWalking</h2>
    <img  src="https://gymnasium.farama.org/_images/cliff_walking.gif" width="700px"/>
    <br>
    Um personagem começa num local fixo e pode andar por um mapa, seu objetivo é chegar até certo local fixo dando a menor quantidade de passos o possível, "cair" em alguns espaços específicos, o personagem é penalizado e retorna ao início do caminho.

  <h2>2.2 - FrozenLake</h2>
    <img  src="https://gymnasium.farama.org/_images/frozen_lake.gif" width="300px"/>
    <br>
    Similarm ao CliffWalking, mas o personagem pode "escorregar", andando perpendicularmente à direção desejada. No entanto, se o personagem cai em um dos buracos, o episódio termina.
    

  <h2>2.3 - Blackjack</h2>
    <img  src="https://gymnasium.farama.org/_images/blackjack.gif" 
    width="300px"/>
    <br>
    Similarmente ao blackjack real, as ações são escolher pegar mais uma carta ou parar, e o objetivo é que o valor total de suas cartas seja mais próximo de 21 que as cartas do Dealer, sem ultrapassar o valor.

  <h2>2.4 - Acrobot</h2>
    <img  src="https://gymnasium.farama.org/_images/acrobot.gif" width="300px"/>
    <br>
    No ambiente Acrobot, o espaço de estados é contínuo e as ações têm espaço discreto, representando a aplicação de torque negativo, positivo, ou neutro nas juntas. O objetivo é fazer com que a extremidade solta da barra ultrapasse a linha na menor quantidade de passos.


<h1>3 - Resultados</h1>
  <h2>Acrobot</h2>
  <img  src="https://github.com/user-attachments/assets/df02cc25-6af8-4a29-b28c-542e22aef2ff" width="700px"/>
  
  <h2>Blackjack</h2>
  <img  src="https://github.com/user-attachments/assets/ed443644-d3c0-43d6-8351-aedb4b6e8485" width="700px"/>

  <h2>CliffWalking</h2>
  <img  src="https://github.com/user-attachments/assets/d99d41cd-d6f8-412e-b5c3-7cdff5f676ea" width="700px" />
    
  <h2>FrozenLake</h2>
  <img  src="https://github.com/user-attachments/assets/fbfc36b3-836c-4343-9317-ce693a4e632c" width="700px"/>



<h1>Observações finais</h1>
 O TD Learning Off-Policy funciona bem em ambientes discretos e estruturados (como FrozenLake e Blackjack), mas pode ter dificuldades em cenários com grandes penalizações e estratégias de navegação mais complexas. Por exemplo no Acrobot o aprendizado demorou um pouco para começar.

Ajustes no ε-decay, nas recompensas e na exploração podem melhorar o desempenho em alguns casos.
  

















      
