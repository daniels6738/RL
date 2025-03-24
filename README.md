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
    A correção no TD Learning off-policy é feita através de métodos como o Ordinary Importance Sampling e o Weighted Importance Sampling.
    
  <h2>Exemplos Simples</h2>
    Vamos ver exemplos de como essa correção é calculada em episódios pequenos, de 1 até 3 passos.
    
  <h3>Ordinary Importance Sampling</h3>
    Para  n=1 :
      experiência:  s,a,r1,s1,a1 
      estimativa:  Qtarget=r1+γ.Q(s1,a1)∗π(a1|s1)/b(a1|s1)

  <h3>Weighted Importance Sampling</h3>
    Para  n=2 :
      experiência:  s,a,r1,s1,a1,r2,s2,a2 
      estimativa:  Qtarget=r1+γ.r2+γ2.Q(s2,a2)∗π(a1|s1)π(a2|s2)/b(a1|s1)b(a2|s2)

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
  

















      
