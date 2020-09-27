// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.pesosCamada = [];
     this.bias = -1;
     this.u =0;
     this.numEpoca = 0;
     this.showEpocas = 0;
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.eqm = 9999999999999999999999999999999999999999999999999; //eqm anterior
     this.entradaCamada1 = [];
     this.entradaCamada2 = [];
     this.entradaCamada3 = [];
     this.saidaCamada1 = [];
     this.saidaCamada2 = [];
     this.saidaCamada3 = [];
     this.gradienteCamada1 = [];
     this.gradienteCamada2 = [];
     this.gradienteCamada3 = [];
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
          var pesos = [];
          for(k=0;k<3;k++){
               for(i=0;i<entradas;i++){
                    pesos[i] = Math.random()/2;
                    console.log("Camada: "+k+" wi"+i+": "+pesos[i]);
               }
               redeNeural.pesosCamada[k] = pesos;
          }
          console.log("Pesos iniciais:");
          console.log(redeNeural.pesosCamada[0]);
     }
     this.recalcularPesos = function(gradiente, inputs, camada){
          var pesos = redeNeural.pesosCamada[camada];
          for(j=0;j<pesos.length;j++){
               let momentum = 0;
               pesos[j] = pesos[j] + momentum + redeNeural.txAprendizagem * gradiente * inputs[j];
          }
          redeNeural.pesosCamada[camada] = pesos;
     }
     this.calcularU = function(inputs,camada){
          var pesos = redeNeural.pesosCamada[camada]; // falhando aqui com vetor undefined após rodar 12 ciclos
          console.log(redeNeural.pesosCamada[camada]);
          var soma = 0;
          for(j=0;j<inputs.length;j++){
               soma += inputs[j] * pesos[j];
          }
          return soma += redeNeural.bias;
     }
     this.calcularEqm = function(dados,camada){
          var eqm = 0;
          var amostras = dados.length;
          for(i=0;i<amostras;i++){
               var u = redeNeural.calcularU(dados[i].inputs,camada);
               eqm += eqm  + ((dados[i].output-u)**2)/2;
          }
          eqm = (eqm/amostras);
          console.log("Eqm(fi): "+eqm);
          return eqm;
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
                    for(i=0;i<qtdEntradas;i++){
                         redeNeural.saidaCamada1[i] = redeNeural.funcaoAtivacao(redeNeural.calcularU(dados[k].inputs,0));
                         redeNeural.saidaCamada2[i] = redeNeural.funcaoAtivacao(redeNeural.calcularU(redeNeural.saidaCamada1[i],1));
                         redeNeural.saidaCamada3[i] = redeNeural.funcaoAtivacao(redeNeural.calcularU(redeNeural.saidaCamada2[i],2));
                    }
                    for(i=0;i<qtdEntradas;i++){
                         redeNeural.gradienteCamada3[i] = (dados[k].output-redeNeural.saidaCamada3[i])*redeNeural.funcaoAtivacao(redeNeural.saidaCamada3[i]);
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada3[i],redeNeural.saidaCamada3[i],2);
                         redeNeural.gradienteCamada2[i] = (dados[k].output-redeNeural.saidaCamada2[i])*redeNeural.funcaoAtivacao(redeNeural.saidaCamada2[i]);
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada2[i],redeNeural.saidaCamada2[i],2);
                         redeNeural.gradienteCamada1[i] = (dados[k].output-redeNeural.saidaCamada1[i])*redeNeural.funcaoAtivacao(redeNeural.saidaCamada1[i]);
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada1[i],redeNeural.saidaCamada1[i],2);
                    }
               }
               console.log(redeNeural.gradienteCamada1);
               numEpoca++;
               eqmAtual = redeNeural.calcularEqm(dados,3);
          }
          redeNeural.showEpocas = numEpoca;
          console.log("Epocas: "+numEpoca);
          console.log("Pesos Finais:");
          for(z=0;z<3;z++){
               console.log("wf"+z+": "+redeNeural.pesosCamada[z]);
          }
     }
}

