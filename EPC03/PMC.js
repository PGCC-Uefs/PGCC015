// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.numEpoca = 0;
     this.showEpocas = 0;
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.camada = [];
     this.iniciar = function(txAprendizagem,numEpoca){
          redeNeural.txAprendizagem=txAprendizagem;
          redeNeural.numEpoca=numEpoca;
     }
     this.iniciaPesos = function(entradas){
          console.log("Pesos iniciais:");
          for(k=0;k<3;k++){
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
     this.calcularU = function(dados,n){
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
     this.calcularEqm = function(amostras){
          var eq,soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = amostras.length;
          alert("EQM: "+JSON.stringify(redeNeural.camada[n]));
          for(i=0;i<qtd;i++){
               soma += amostras[i].output-redeNeural.camada[n].outputsU;
               eq += eq  + (soma**2)/2;
          }
          eq = (eq/qtd);
          return eq;
     }
     this.executar = function(inputs,n){
          var u = redeNeural.calcularU(inputs,n);
          return redeNeural.funcaoAtivacao(u);
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 9999999999999999999999999999999999999999;
          var eqmAtual = 0;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<dados.length;k++){
                    for(n=0;n<3;n++){
                         redeNeural.calcularU(dados[k].inputs,n)
                         redeNeural.funcaoAtivacao(n);
                    }
                    for(n=2;n>=0;n--){
                         redeNeural.calculaGradiente(dados[k].output,n);
                         redeNeural.recalcularPesos(n);
                    }
               }
               eqmAtual = redeNeural.calcularEqm(dados);
               alert(eqmAtual);
               numEpoca++;
          }
          redeNeural.showEpocas = numEpoca;
          console.log("Epocas: "+numEpoca);
     }
}

