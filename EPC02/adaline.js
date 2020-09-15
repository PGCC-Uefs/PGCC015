// Mestrado PGCC Uefs 2020.1
// rede neural Adaline EPC02
// Noberto Pires Maciel

console.log("adaline:");

this.adaline = function(){
     var redeNeural = this;
     this.pesos = [];
     this.bias = -1;
     this.numEpoca = 1000; //valor default
     this.showEpocas = 0;
     this.txAprendizagem = 0.0025; //valor default
     this.precisao = 10**-6;
     this.eqm = 0; //eqm anterior
     this.funcaoAtivacao = function(u){
          //return 1 / (1 + Math.exp(-1 * u)); // sigmoid
          return (u >= 0? 1: -1); // sinal
     }
     this.iniciar = function(txAprendizagem,numEpoca){
          redeNeural.txAprendizagem=txAprendizagem;
          redeNeural.numEpoca=numEpoca;
     }
     this.iniciaPesos = function(entradas){
          console.log("Pesos iniciais:");
          for(i=0;i<entradas;i++){
                redeNeural.pesos[i] = Math.random();
                console.log("wi"+i+": "+redeNeural.pesos[i]); // apresenta na tela
          }
     }
     this.recalcularPesos = function(diferenca, inputs){
          for(j=0;j<redeNeural.pesos.length;j++){
             redeNeural.pesos[j] = redeNeural.pesos[j] + redeNeural.txAprendizagem * diferenca * inputs[j];
          }
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
               eqm += eqm  + (dados[i].output-u)**2;
          }
          eqm = (eqm/amostras);
          //console.log("Eqm(fi): "+eqm);
          return eqm;
     }
/*      this.calcularEqm = function(dados){
          redeNeural.eqm = 0;
          var amostras = dados.length;
          for(i=0;i<amostras;i++){
               var u = redeNeural.calcularU(dados[i].inputs);
               eqmAtual += (dados[i].output-u);
          }
          redeNeural.eqm = ((eqmAtual**2)/amostras) - redeNeural.eqm; // eqm anterior EQM(t), t=epoca
          console.log("Eqm(fi): "+redeNeural.eqm+" Eqm(at): "+eqmAtual);
     } */
     this.executar = function(inputs){
          var u = redeNeural.calcularU(inputs);
          return redeNeural.funcaoAtivacao(u);
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 0;
          var eqmAtual = 1;
          var tamPesos = dados[0].inputs.length;
          redeNeural.iniciaPesos(tamPesos);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               var diferenca = 0;
               eqmAnterior = eqmAtual;
               for(i=0;i<dados.length;i++){
                    var u = redeNeural.calcularU(dados[i].inputs);
                    diferenca = dados[i].output - u;
                    redeNeural.recalcularPesos(diferenca, dados[i].inputs);
               }
               numEpoca++;
               eqmAtual = redeNeural.calcularEqm(dados);
          }
          // apresenta na tela
          redeNeural.showEpocas = numEpoca;
          console.log("Pesos Finais:");
          for(z=0;z<tamPesos;z++){
               console.log("wf"+z+": "+redeNeural.pesos[z]);
          }
          console.log("u: "+u);
     }
}
