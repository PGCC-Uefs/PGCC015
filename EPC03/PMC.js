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
     this.calcularU = function(dados,camada){
          var neuronios = redeNeural.camada[camada].neuronios;
          var arraySaida = [];
          for(i=0;i<neuronios.length;i++){
               var soma = 0;
               var pesos = neuronios[i].pesos;
               for(j=0;j<pesos.length;j++){
                    soma += dados[j] * pesos[j];
               }
               arraySaida[i] = soma += redeNeural.bias;
/*                alert(
                    "# CalcularU #\n"+
                    "Neurônio: "+i+
                    "\nSoma: "+soma+
                    "\ncamada:"+camada+
                    "\nPesos: "+JSON.stringify(pesos)+
                    "\nSaida camada 0: "+JSON.stringify(redeNeural.camada[0].outputs)+
                    "\nSaida camada 1: "+JSON.stringify(redeNeural.camada[1].outputs)+
                    "\nSaida camada 2: "+JSON.stringify(redeNeural.camada[2].outputs)
               ); */
          }
          redeNeural.camada[camada].outputs = arraySaida;
     }
     this.funcaoAtivacao = function(camada){
          var neuronios = redeNeural.camada[camada];
          var ativacao = [];
          for(i=0;i<neuronios.outputs.length;i++){
               ativacao[i] = 1*(-Math.exp(-1 * neuronios.outputs[i]))/(1 + Math.exp(-1 * neuronios.outputs[i]))**2;
          }
          redeNeural.camada[camada].outputs = ativacao;
     }
     this.calculaGradiente = function(output,camada){
          var gradiente = [];
          var neuronios = redeNeural.camada[camada];
          for(i=0;i<neuronios.outputs.length;i++){
               gradiente[i] = (output-neuronios.outputs[i])*neuronios.outputs[i]; //gradiente saida 3 = -(dj³-Yj³)*g'(Ij³)
          }
          redeNeural.camada[camada].gradiente = gradiente; // aqui é para substituir a chave e está apagando todo o neurônio
     }
     this.recalcularPesos = function(camada){
          var neuronios = redeNeural.camada[camada];
          for(i=0;i<neuronios.length;i++){
               var pesos = neuronios[i].pesos;
               for(j=0;j<pesos.length;j++){
                    let momentum = 0; // INSERIR EQUAÇÃO
                    pesos[j] = pesos[j] + momentum + redeNeural.txAprendizagem /* * neuronios.gradiente[i] */ /* rever este gradiente no array  */ * neuronios.outputs[i];
               }
               neuronios[i] = {pesos: pesos};
          }
          redeNeural.camada[camada].neuronios = neuronios; // aqui é para substituir a chave e está apagando todo o neurônio
     }
     this.calcularEqm = function(dados){
          var eq = 0;
          var amostras = redeNeural.camada[2].length;
          for(i=0;i<amostras.length;i++){
               var u = redeNeural.calcularU(amostras[i].outputs,2);
               eq += eq  + ((dados[i].output-u)**2)/2;
          }
          eq = (eq/amostras);
          return eq;
     }
     this.executar = function(inputs,camada){
          var u = redeNeural.calcularU(inputs,camada);
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
                    for(camada=0;camada<3;camada++){
                         redeNeural.calcularU(dados[k].inputs,camada)
                         alert("U:\nCamada: "+camada+" \nneuronios: "+JSON.stringify(redeNeural.camada[camada]));
                         redeNeural.funcaoAtivacao(camada);
                         alert("Ativação:\nCamada: "+camada+" \nneuronios: "+JSON.stringify(redeNeural.camada[camada]));
                    }
                    for(camada=2;camada>-1;camada--){
                         redeNeural.calculaGradiente(dados[k].output,camada);
                         alert("Gradiente:\nCamada: "+camada+" \nneuronios: "+JSON.stringify(redeNeural.camada[camada]));
                         redeNeural.recalcularPesos(camada);
                         alert("Recálculo pesos:\nCamada: "+camada+" \nneuronios: "+JSON.stringify(redeNeural.camada[camada]));
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

