// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.amostras = 0; // quantidade de amostras 
     this.numEpocas = 10**6;
     this.eq = [];
     this.topologia = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-3;
     this.alfa = 0.1;
     this.camada = [];
     this.numCamadas = [];
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
                    var pesos = [];
                    for(i=0;i<qtdEntradas;i++){
                         pesos[i] = Math.random()/2;
                         console.log("Camada: "+k+" wi"+i+": "+pesos[i]);
                    }
                    neuronios[j] = {pesos: pesos};
               }
               redeNeural.camada[k] = {neuronios: neuronios};
          }
     }
     this.calcularU = function(dados,n){
          var camada = redeNeural.camada[n].neuronios;
          var arraySaida = [];
          for(i=0;i<camada.length;i++){
               var soma = 0;
               var pesos = camada[i].pesos;
               for(j=0;j<pesos.length;j++){
                    soma += dados[j] * pesos[j];
               }
               (redeNeural.camada.length < n? arraySaida[i] = soma + redeNeural.bias : arraySaida[i] = soma);
          }
          redeNeural.camada[n].outputsU = arraySaida;
          document.getElementById("dadosSaida").append("\ncamada: "+n+"\nU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\n\n");
     }
     this.funcaoAtivacao = function(n){
          var camada = redeNeural.camada[n];
          var ativacao = [];
          var bias = redeNeural.bias;
          for(i=0;i<camada.outputsU.length;i++){
               ativacao[i] = 1/(1 + Math.exp(-bias * camada.outputsU[i])); // VER ESSE bias AQUI
          }
          (n<redeNeural.camada.length-1)?ativacao.unshift(bias):0;
          redeNeural.camada[n].outputs = ativacao;
          document.getElementById("dadosSaida").append("\ncamada: "+n+"\nativação: "+JSON.stringify(redeNeural.camada[n])+"\n\n");
     }
     this.derivada = function(n){
          var outputs = redeNeural.camada[n].outputs;
          var derivada = [];
          var b = redeNeural.bias;
          for(i=0;i<outputs.length;i++){
               derivada[i] = b*outputs[i]*(1-outputs[i]);
          }
          redeNeural.camada[n].derivada = derivada;
     }
     this.calculaGradiente = function(output,n){
          var gradiente = [];
          var camada = redeNeural.camada[n];
          for(i=0;i<camada.outputs.length;i++){
               gradiente[i] = (output-camada.outputs[i])*camada.derivada[i]; //gradiente saida 3 = -(dj³-Yj³)*g'(Ij³)
          }
          redeNeural.camada[n].gradiente = gradiente;
          document.getElementById("dadosSaida").append("\ncamada: "+n+"\ngradiente: "+JSON.stringify(redeNeural.camada[n])+"\n\n");
     }
     this.recalcularPesos = function(inputs,n){
          var camada = redeNeural.camada[n];
          for(i=0;i<camada.neuronios.length;i++){
               var saida = [];
               var pesos = camada.neuronios[i].pesos;
               (n==0? saida = inputs : saida = redeNeural.camada[n-1].outputs);
               for(j=0;j<saida.length;j++){
                    let alfa = redeNeural.alfa;
                    let ngy = pesos[j] + redeNeural.txAprendizagem * camada.gradiente[i] * saida[i];
                    let momentum = alfa*(ngy - pesos[j]);
                    pesos[j] = ngy + momentum;
               }
               camada.neuronios[i] = {pesos: pesos};
          }
          document.getElementById("dadosSaida").append("\ncamada "+n+"\nqtdSaidas: "+saida.length+"\nrecalcularPesos: "+JSON.stringify(camada.neuronios)+"\n\n");
     }
     this.eqRede = function(dK,k){
          var eqRede = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = redeNeural.camada[n].outputsU.length;
          for(i=0;i<qtd;i++){
               soma += dK-redeNeural.camada[n].outputsU[i];
          }
          eqRede = (soma**2)/2;
          document.getElementById("dadosSaida").append("\nAMOSTRA: "+k+"\ncamada: "+n+"\neqRede: "+JSON.stringify(eqRede)+"\n\n");
          return eqRede;
     }
     this.calcularEqm = function(){
          var soma = 0;
          var p = redeNeural.amostras ;
          for(i=0;i<p;i++){
               soma += redeNeural.eq[i];
          }
          return soma/p;
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 999999999;
          var eqmAtual = 0;
          var camadas = redeNeural.numCamadas;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.amostras = dados.length;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<redeNeural.amostras ;k++){
                    for(n=0;n<camadas;n++){
                         redeNeural.calcularU(dados[k].inputs,n);
                         redeNeural.funcaoAtivacao(n);
                         redeNeural.derivada(n);
                    }
                    for(m=(camadas-1);m>=0;m--){
                         redeNeural.calculaGradiente(dados[k].output,m);
                         redeNeural.recalcularPesos(dados[k].inputs,m);
                    }
                    redeNeural.eq[k] = redeNeural.eqRede(dados[k].output,k);
               }
               eqmAtual = redeNeural.calcularEqm();
               numEpoca++;
          }
          redeNeural.numEpocas = numEpoca;
          redeNeural.tempoExecucao[1] = Date.now();
          redeNeural.tempoExecucao[2] = redeNeural.tempoExecucao[1]-redeNeural.tempoExecucao[0];
          document.getElementById("tempo").innerText = redeNeural.tempoExecucao[2];
          document.getElementById("erro").innerText = Math.abs(eqmAtual - eqmAnterior);
          document.getElementById("eqm").innerText = eqmAtual;
          document.getElementById("epocas").innerText = numEpoca;
          console.log("Tempo de execução: "+redeNeural.tempoExecucao[2]);
          console.log("EQM Total: "+eqmAtual);
          console.log("Erro: "+Math.abs(eqmAtual - eqmAnterior));
          console.log("Amostras: "+k);
          console.log("Epocas: "+numEpoca);
     }
     this.executar = function(dados){
          var camadas = redeNeural.numCamadas;
          for(k=0;k<dados.length;k++){
               for(n=0;n<camadas;n++){
                    redeNeural.calcularU(dados[k].inputs,n)
                    redeNeural.funcaoAtivacao(n);
               }
               console.log(redeNeural.camada[camadas-1].outputs);
          }
          return redeNeural.camada[camadas-1].outputs;
     }
}

