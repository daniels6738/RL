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


  <h2>Vamos implementar as variantes do off-policy:</h2>
    <ul>
      <li>Ordinary Importance Sampling</li>
      <li>Weighted Importance Sampling</li>
      <li>Pesos Truncados</li>
    </ul>
    
<h1>2 - Lidando com Estados Contínuos</h1>

<h1>3 - Otimizando Parâmetros</h1>
  <h2>3.1 - Ambiente Discreto</h2>
  <h2>3.2 - Ambiente Continuo</h2>

<h1>4 - Experimentos Completos</h1>
  <h2>Racetrack</h2>
  <h2>CartPole</h2>

<h1>Cliffwalking</h1>

<h1>Observações finais</h1>
  

















      
