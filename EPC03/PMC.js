// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.k = 1;               // declividade
     this.x0 = 0;              // ponto médio da curva sigmoide
     this.l = 1;
     this.amostras = []; 
     this.numEpocas = 10**6;
     this.eq = [];
     this.topologia = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.e = 2.718281828;
     this.alfa = 0.5; // alfa do termo de momentum
     this.sinalGradiente = -1;
     this.camada = [];
     this.wbefore = [];
     this.numCamadas = 0;
     this.classes = [];
     this.tempoExecucao = [];
     this.iniciar = function(txAprendizagem,topologia,classes){
          redeNeural.txAprendizagem=txAprendizagem;
          redeNeural.classes = classes;
          redeNeural.topologia = topologia;
          redeNeural.numCamadas = redeNeural.topologia.length;
          redeNeural.tempoExecucao[0] = Date.now();
          console.log("Tempo de início: "+redeNeural.tempoExecucao[0]+" ms");
          console.log("Topologia: "+JSON.stringify(topologia));
     }
     this.iniciaPesos = function(qtdEntradas){
          console.log("Pesos iniciais:");
          var camadas = redeNeural.numCamadas;
          for(k=0;k<camadas;k++){
               var neuronios = [];
               var topologia = redeNeural.topologia[k];
               for(j=0;j<topologia;j++){
                    (k == 0? entradas = qtdEntradas : entradas = redeNeural.topologia[k-1]);
                    var pesos = [];
                    for(i=0;i<entradas;i++){
                         pesos[i] = Math.random()/2;
                         console.log("Camada: "+k+" wi"+i+": "+pesos[i]);
                    }
                    neuronios[j] = {pesos: pesos};
               }
               redeNeural.camada[k] = {neuronios: neuronios, wbefore: neuronios};
          }
     }
     this.calcularU = function(dados,n){
          var camada = redeNeural.camada[n].neuronios;
          var qtdNeuronios = camada.length;
          var arraySaida = [];
          if(n>0){dados = redeNeural.camada[n-1].outputs};
          console.log("U camada, pessos geral: "+JSON.stringify(camada));
          for(i=0;i<qtdNeuronios;i++){
               var soma = 0;
               var pesos = camada[i].pesos;
               console.log("U: pesos-> "+JSON.stringify(pesos));
               for(j=0;j<pesos.length;j++){
                    soma += dados[j] * pesos[j];
               }
               (n==redeNeural.camada.length-1? arraySaida[i] = soma : arraySaida[i] = soma + redeNeural.bias);
          }
          redeNeural.camada[n].outputsU = arraySaida;
          console.log("U saída camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].outputsU));
          console.log("U pesos camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.funcaoAtivacao = function(n){
          var neuronios = redeNeural.camada[n];
          var ativacao = [];
          var k = redeNeural.k;               // declividade
          var x0 = redeNeural.x0;              // ponto médio da curva sigmoide
          var l = redeNeural.l;               // valor máximo da curva
          var e = redeNeural.e;    // número neperiano
          for(i=0;i<neuronios.outputsU.length;i++){
               ativacao[i] = l/(1 + e*Math.exp(-k*(neuronios.outputsU[i]-x0)));
          }
          redeNeural.camada[n].outputs = ativacao;
          console.log(JSON.stringify("Funçao ativação camada: "+n+" -> "+redeNeural.camada[n].outputs));
          console.log("Função ativação pesos camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.dSigmoid = function(n){
          var outputs = redeNeural.camada[n].outputs;
          var dSigmoid = [];
          var k = redeNeural.k;               // declividade
          var l = redeNeural.l;               // valor máximo da curva
          var e = redeNeural.e;              // número neperiano
          for(i=0;i<outputs.length;i++){
               dSigmoid[i] = k*(e**-k*outputs[i])/(((e**-k*outputs[i])+1)**2); //(e**-outputs[i])*(l+(e**-outputs[i]))**2; //outputs[i]*(1-outputs[i]); //outputs[i]*(1-outputs[i]);
          }
          redeNeural.camada[n].dSigmoid = dSigmoid;
          console.log("dSigmoid pesos camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.calcularGradiente = function(k,n){
          var gradiente = [];
          var sinal = redeNeural.sinalGradiente;
          var outputAmostra = redeNeural.amostras[k].output.split(""); // transformado em array
          var camada = redeNeural.camada[n];
          for(i=0;i<outputAmostra.length;i++){
               let dK = outputAmostra[i];
               let output = camada.outputs[i];
               let dSigmoid = camada.dSigmoid[i];
               gradiente[i] = sinal*(dK-output)*dSigmoid;//sinal*(output-camada.dSigmoid[i]);
          }
          redeNeural.camada[n].gradiente = gradiente;
          console.log("Gradiente: "+n+" -> "+JSON.stringify(redeNeural.camada[n].gradiente));
     }
     this.recalcularPesos = function(k,n){
          var camada = redeNeural.camada[n];
          var inputs = redeNeural.amostras[k].inputs
          var qtdNeuronios = camada.neuronios.length;
          for(i=0;i<qtdNeuronios;i++){
               var saida = [];
               var wbefore = camada.wbefore[i].pesos;
               var pesos = camada.neuronios[i].pesos;
               camada.wbefore[i].pesos = pesos;
               (n>0? saida = redeNeural.camada[n-1].outputs : saida = inputs);
               for(j=0;j<pesos.length;j++){
                    let alfa = redeNeural.alfa;
                    let ngy = pesos[j] + redeNeural.txAprendizagem * camada.gradiente[i] * saida[i];
                    let momentum = alfa*(ngy - wbefore[j]); // ngy = txAprendizado*gradeiante*saída Y
                    pesos[j] = ngy + momentum;
/*                     alert("TESTE DE MESA"+
                    "\nj (pesos.length): "+j+
                    "\npesos length: "+JSON.stringify(pesos.length)+
                    "\nk (amostra): "+k+
                    "\nn (camada): "+n+
                    "\nInputs: "+inputs.length+
                    "\npesos: "+JSON.stringify(pesos)+
                    "\nSaida length: "+JSON.stringify(saida.length)+
                    "\nSaida: "+JSON.stringify(saida)+
                    "\nngy: "+JSON.stringify(ngy)+
                    "\nwBefore: "+JSON.stringify(wbefore[j])+
                    "\nmomentum: "+JSON.stringify(momentum)+
                    "\npesos: "+JSON.stringify(pesos[j])); */
               }
               camada.neuronios[i].pesos = pesos;
               console.log("Pesos gerais\ncamada: "+n+"\nneuronio: "+i+"\n"+JSON.stringify(camada));
          }
          console.log("wBefore: "+n+" -> "+JSON.stringify(camada.wbefore));
     }
     this.eqAmostra = function(dK){
          var eqAmostra = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = redeNeural.camada[n].outputs.length;
          for(i=0;i<qtd;i++){
               soma += dK[i]-redeNeural.camada[n].outputs[i];
          }
          eqAmostra = (soma**2)/2;
          return eqAmostra;
     }
     this.calcularEqm = function(){
          var soma = 0;
          var eqm = 0;
          var p = redeNeural.amostras.length;
          for(i=0;i<p;i++){
               soma += redeNeural.eq[i];
          }
          eqm =  soma/p;
          return eqm;
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 99999999999999999999999999999999999999999999;
          var eqmAtual = 0;
          var camadas = redeNeural.numCamadas;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.amostras = dados;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<redeNeural.amostras.length;k++){
                    var input = dados[k].inputs;
                    for(n=0;n<camadas;n++){
                         console.log("camada laço treinamento: "+n+"\ninput: "+input);
                         redeNeural.calcularU(input,n);
                         redeNeural.funcaoAtivacao(n);
                         redeNeural.dSigmoid(n);
                         input = redeNeural.camada[n].outputs;
                         if(n<(camadas-1)){input.unshift(redeNeural.bias);}
                         alert("Treinamento \ncamada: "+n+"\nneuronio: "+i+"\n"+JSON.stringify(redeNeural.camada[n].neuronios));
                    }
                    redeNeural.eq[k] = redeNeural.eqAmostra(dados[k].output);
                    for(m=(camadas-1);m>=0;m--){
                         redeNeural.calcularGradiente(k,m);
                         redeNeural.recalcularPesos(k,m);
                    }
               }
               eqmAtual = redeNeural.calcularEqm();
               numEpoca++;
          }
          redeNeural.numEpocas = numEpoca;
          redeNeural.tempoExecucao[1] = Date.now();
          redeNeural.tempoExecucao[2] = redeNeural.tempoExecucao[1]-redeNeural.tempoExecucao[0];
          document.getElementById("saida").innerText = JSON.stringify(redeNeural.camada[redeNeural.topologia.length-1].outputs);
          document.getElementById("tempo").innerText = redeNeural.tempoExecucao[2];
          document.getElementById("erro").innerText = Math.abs(eqmAtual - eqmAnterior);
          document.getElementById("eqm").innerText = eqmAtual;
          document.getElementById("epocas").innerText = numEpoca;
          console.log("Tempo de execução: "+redeNeural.tempoExecucao[2]);
          console.log("EQM Total: "+eqmAtual);
          console.log("Erro: "+Math.abs(eqmAtual - eqmAnterior));
          console.log("Amostras: "+(k-1));
          console.log("Epocas: "+numEpoca);
     }
     this.executar = function(dados){
          var numCamadas = redeNeural.numCamadas;
          var resultado = [];
          for(k=0;k<dados.length;k++){
               var inputs = dados[k].inputs;
               for(n=0;n<numCamadas;n++){
                    redeNeural.calcularU(inputs,n);
                    redeNeural.funcaoAtivacao(n);
                    console.log(JSON.stringify(redeNeural.camada[numCamadas-1].outputsU));
               }
               resultado[k] = redeNeural.camada[numCamadas-1].outputsU;
               console.log(JSON.stringify(inputs));
               console.log(JSON.stringify(resultado[k]));
          }
     }

}

