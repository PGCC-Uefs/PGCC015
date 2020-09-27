// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.camadas = [];
     this.pesos = [];
     this.bias = -1;
     this.u =0;
     this.numEpoca = 0;
     this.showEpocas = 0;
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.eqm = 9999999999999999999999999999999999999999999999999; //eqm anterior
     this.funcaoAtivacao = function(u){
          // return 1 / (1 + Math.exp(-1 * u)); // sigmoid logística, AJUSTAR
          // derivada 1'*(1 + Math.exp(-1 * u))-1*(1 + Math.exp(-1 * u))'/(1 + Math.exp(-1 * u))^2
          // derivada 0 *(1 + Math.exp(-1 * u))-1*(-Math.exp(-1 * u))/(1 + Math.exp(-1 * u))^2
           return 1*(-Math.exp(-1 * u))/(1 + Math.exp(-1 * u))**2;
          // redeNeural.u = u;
          // return (u >= 0? 1: -1); // sinal
     }
     this.iniciar = function(txAprendizagem,numEpoca){
          redeNeural.txAprendizagem=txAprendizagem;
          redeNeural.numEpoca=numEpoca;
     }
     this.iniciaPesos = function(entradas){
          console.log("Pesos iniciais:");
          for(i=0;i<entradas;i++){
                redeNeural.pesos[i] = Math.random()/2;
                console.log("wi"+i+": "+redeNeural.pesos[i]);
          }
     }
     this.recalcularPesos = function(gradiente, inputs, camada){
          var pesos = [];
          console.log(inputs);
          for(j=0;j<redeNeural.pesos.length;j++){
               let momentum = 0;
               pesos[j] = redeNeural.pesos[j] + momentum + redeNeural.txAprendizagem * gradiente * inputs[j];
          }
          redeNeural.camadas[camada] = pesos;
     }
     this.calcularU = function(inputs){
          var soma = 0;
          for(j=0;j<inputs.length;j++){
               soma += inputs[j] * redeNeural.pesos[j];
          }
          return soma += redeNeural.bias;
     }
     this.calcularEqm = function(dados){
          var eqm = 0;
          var amostras = dados.length;
          for(i=0;i<amostras;i++){
               var u = redeNeural.calcularU(dados[i].inputs);
               eqm += eqm  + ((dados[i].output-u)**2)/2;
          }
          eqm = (eqm/amostras);
          console.log("Eqm(fi): "+eqm);
          return eqm;
     }
     this.executar = function(inputs){
          var u = redeNeural.calcularU(inputs);
          return redeNeural.funcaoAtivacao(u);
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 9999999999999999999999999999999999999999;
          var eqmAtual = 0;
          var tamPesos = dados[0].inputs.length;
          redeNeural.iniciaPesos(tamPesos);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<dados.length;k++){
                    // tem erro, deve calcular a saída do neuronio 1 para vários inputs*pesos
                    let i1 = redeNeural.calcularU(dados[k].inputs); // I = Σ(X*W)
                    let y1 = redeNeural.funcaoAtivacao(i1);

                    let i2 = redeNeural.calcularU(y1);
                    let y2 = redeNeural.funcaoAtivacao(i2);
                    
                    let i3 = redeNeural.calcularU(y2);
                    let y3 = redeNeural.funcaoAtivacao(i3);
                    
                    let gradiente3 = (dados[k].output-y3)*redeNeural.funcaoAtivacao(i3);
                    redeNeural.recalcularPesos(gradiente3,y3,2); // aqui está enviando y3=inputs como number e não como array
                    
                    let gradiente2 = (dados[k].output-y2)*redeNeural.funcaoAtivacao(i2);
                    redeNeural.recalcularPesos(gradiente2,y2,1);
                    
                    let gradiente1 = (dados[k].output-y1)*redeNeural.funcaoAtivacao(i1);
                    redeNeural.recalcularPesos(gradiente1,y1,0);
               }
               numEpoca++;
               eqmAtual = redeNeural.calcularEqm(dados);
          }
          redeNeural.showEpocas = numEpoca;
          console.log("Epocas: "+numEpoca);
          console.log("Pesos Finais:");
          for(z=0;z<tamPesos;z++){
               console.log("wf"+z+": "+redeNeural.pesos[z]);
          }
          console.log(redeNeural.camadas);
     }
}

