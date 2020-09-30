// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.p = 0; // quantidade de amostras
     this.numEpocas = 0;
     this.eqRede = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.camada = [];
     this.numCamadas = [];
     this.classes = [];
     this.iniciar = function(txAprendizagem,numCamadas,classes){
          redeNeural.txAprendizagem=txAprendizagem;
          redeNeural.classes = classes;
          redeNeural.numCamadas = numCamadas;
     }
     this.iniciaPesos = function(entradas){
          var camadas = redeNeural.numCamadas;
          console.log("Pesos iniciais:");
          for(k=0;k<camadas;k++){
               var neuronios = [];
               for(j=0;j<entradas;j++){
                    var pesos = [];
                    for(i=0;i<entradas;i++){
                         pesos[i] = Math.random()/2;
                         console.log("Camada: "+k+" wi"+i+": "+pesos[i]);
                    }
                    neuronios[j] = {pesos: pesos};
               }
               redeNeural.camada[k] = {neuronios: neuronios};
          }
     }
     this.calcularU = function(dados,n){ // falhando, neuronio indefinido no array, ajuste em redeNeural.executar()
          var camada = redeNeural.camada[n].neuronios;
          var arraySaida = [];
          for(i=0;i<camada.length;i++){
               var soma = 0;
               var pesos = camada[i].pesos;
               for(j=0;j<pesos.length;j++){
                    soma += dados[j] * pesos[j];
               }
               arraySaida[i] = soma += redeNeural.bias;
          }
          redeNeural.camada[n].outputsU = arraySaida;
     }
     this.funcaoAtivacao = function(n){
          var camada = redeNeural.camada[n];
          var ativacao = [];
          console.log(n);
          for(i=0;i<camada.outputsU.length;i++){
               ativacao[i] = 1*(-Math.exp(-1 * camada.outputsU[i]))/(1 + Math.exp(-1 * camada.outputsU[i]))**2;
          }
          redeNeural.camada[n].outputs = ativacao;
     }
     this.calculaGradiente = function(output,n){
          var gradiente = [];
          var camada = redeNeural.camada[n];
          for(i=0;i<camada.outputs.length;i++){
               gradiente[i] = (output-camada.outputs[i])*camada.outputs[i]; //gradiente saida 3 = -(dj³-Yj³)*g'(Ij³)
          }
          redeNeural.camada[n].gradiente = gradiente;
     }
     this.recalcularPesos = function(n){
          var camada = redeNeural.camada[n];
          for(i=0;i<camada.neuronios.length;i++){
               var pesos = camada.neuronios[i].pesos;
               for(j=0;j<pesos.length;j++){
                    let momentum = 0; // INSERIR EQUAÇÃO
                    pesos[j] = pesos[j] + momentum + redeNeural.txAprendizagem * camada.gradiente[i] * camada.outputs[i];
               }
               camada.neuronios[i] = {pesos: pesos};
          }
     }
     this.eqRede = function(dK){
          var eqRede = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = redeNeural.camada[n].outputsU.length;
          for(i=0;i<qtd;i++){
               soma += dK-redeNeural.camada[n].outputsU[i];
          }
          return eqRede = (soma**2)/2;
     }
     this.calcularEqm = function(){
          var soma = 0;
          var camadas = redeNeural.numCamadas;
          var p = redeNeural.p;
          for(i=0;i<p;i++){
               soma += redeNeural.eqRede[i];
          }
          return soma/p;
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 9999999999999999999999999999999999999999;
          var eqmAtual = 0;
          var camadas = redeNeural.numCamadas;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.p = dados.length;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<redeNeural.p;k++){
                    for(n=0;n<camadas;n++){
                         redeNeural.calcularU(dados[k].inputs,n)
                         redeNeural.funcaoAtivacao(n);
                    }
                    for(m=(camadas-1);m>=0;m--){
                         redeNeural.calculaGradiente(dados[k].output,m);
                         redeNeural.recalcularPesos(m);
                    }
                    redeNeural.eqRede[k] = redeNeural.eqRede(dados[k].output);
               }
               eqmAtual = redeNeural.calcularEqm();
               numEpoca++;
               console.log("EQM Total: "+eqmAtual);
               console.log("EqRede: "+redeNeural.eqRede);
          }
          redeNeural.numEpocas = numEpoca;
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
     }
}

