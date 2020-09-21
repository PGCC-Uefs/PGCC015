// Mestrado PGCC Uefs 2020.1
// rede neural Perceptron EPC01
// Noberto Pires Maciel

console.log("PERCEPTRON:");

this.perceptron = function(){
     var redeNeural = this;
     this.pesos = [];
     this.bias = -1;
     this.u =0;
     this.numEpoca = 1000;
     this.showEpocas = 0;
     this.txAprendizagem = 0.01;
     this.funcaoAtivacao = function(u){
          //console.log(1 / (1 + Math.exp(-1 * u))); // sigmoid
          redeNeural.u = u;
          return (u >= 0? 1: -1);
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
     this.executar = inputs => {
          var u = redeNeural.calcularU(inputs);
          return redeNeural.funcaoAtivacao(u);
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var erro = true;
          var tamPesos = dados[0].inputs.length;
          redeNeural.iniciaPesos(tamPesos);
          while(erro && numEpoca < redeNeural.numEpoca){
               erro = false;
               var diferenca = 0;
               for(i=0;i<dados.length;i++){
                    redeNeural.bias = dados[i].inputs[0];
                    var resultado = redeNeural.executar(dados[i].inputs); // roda o teste
                    if(resultado != dados[i].output){
                         erro = true;
                         diferenca = dados[i].output - resultado;
                         redeNeural.recalcularPesos(diferenca, dados[i].inputs);
                    }
                    else{
                         erro = false;
                    }
               }
               numEpoca++;
          }
          // apresenta na tela
          redeNeural.showEpocas = numEpoca;
          console.log("Pesos Finais:");
          for(z=0;z<tamPesos;z++){
               console.log("wf"+z+": "+redeNeural.pesos[z]);
          }
     }
}
