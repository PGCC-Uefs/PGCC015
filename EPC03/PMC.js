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
     this.saidaCamada0 = [];
     this.saidaCamada1 = [];
     this.saidaCamada2 = [];
     this.gradienteCamada0 = [];
     this.gradienteCamada1 = [];
     this.gradienteCamada2 = [];
     this.funcaoAtivacao = function(u){
          var ativacao = [];
          for(i=0;i<u.length;i++){
               ativacao[i] = 1*(-Math.exp(-1 * u[i]))/(1 + Math.exp(-1 * u))**2;
          }
          return ativacao;
     }
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
               redeNeural.pesosCamada[k] = {neuronio: neuronios};
          }
     }
     this.recalcularPesos = function(gradiente, inputs, camada){
          var neuronios = redeNeural.pesosCamada[camada].neuronio;
          for(i=0;i<neuronios.length;i++){
               var pesos = neuronios[i].pesos;
               //console.log(inputs);
               for(j=0;j<pesos.length;j++){
                    let momentum = 0; // fazer formula
                    pesos[j] = pesos[j] + momentum + redeNeural.txAprendizagem * gradiente * inputs[j];
               }
               neuronios[i] = {pesos: pesos};
          }
          redeNeural.pesosCamada[camada] = {neuronio: neuronios};
     }
     this.calcularU = function(inputs,camada){
          var neuronios = redeNeural.pesosCamada[camada].neuronio;
          var arraySaida = [];
          for(i=0;i<neuronios.length;i++){
               var soma = 0;
               var pesos = neuronios[i].pesos;
               for(j=0;j<pesos.length;j++){
                    soma += inputs[j] * pesos[j];
                    alert(
                    "Neurônio: "+i+
                    "\nSoma: "+soma+
                    "\ncamada:"+camada+
                    "\nPesos: "+JSON.stringify(pesos)+
                    "\nSaida camada 0: "+JSON.stringify(redeNeural.saidaCamada0)+
                    "\nSaida camada 1: "+JSON.stringify(redeNeural.saidaCamada1)+
                    "\nSaida camada 2: "+JSON.stringify(redeNeural.saidaCamada2)
                    );
               }
               arraySaida[i] = soma += redeNeural.bias;
          }
          return  {inputs: arraySaida};
     }
     this.eq = function(camada){
          var eq = 0;
          var amostras = redeNeural.saidaCamada1.length;
          alert(amostras);
          for(i=0;i<amostras;i++){
               var u = redeNeural.calcularU(dados[i].inputs,camada);
               eq += eq  + ((dados[i].output-u)**2)/2;
          }
          eq = (eq/amostras);
          console.log("Eqm(fi): "+eq);
          return eq;
     }


     // erro aqui, usando array dados que não existe
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
          var eqmAnterior = 0;
          var eqmAtual = 9999999999999999999999999999999999999999;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               eqmAnterior = eqmAtual;
               for(k=0;k<dados.length;k++){
                         redeNeural.saidaCamada0 = redeNeural.calcularU(dados[k].inputs,0);
                         redeNeural.saidaCamada0 = redeNeural.funcaoAtivacao(redeNeural.saidaCamada0,1);
                         
                         redeNeural.saidaCamada1 = redeNeural.calcularU(redeNeural.saidaCamada0,1);
                         redeNeural.saidaCamada1 = redeNeural.funcaoAtivacao(redeNeural.saidaCamada1,1);
  
                         redeNeural.saidaCamada2 = redeNeural.calcularU(redeNeural.saidaCamada1,2);
                         redeNeural.saidaCamada2 = redeNeural.funcaoAtivacao(redeNeural.saidaCamada2,2);

                    for(i=0;i<qtdEntradas;i++){
                         redeNeural.gradienteCamada2[i] = (dados[k].output-redeNeural.saidaCamada2[i])*redeNeural.saidaCamada2[i];
                         // verificar se aqui entra a função de ativação recalculando a saída
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada2[i],redeNeural.saidaCamada2[i],2);
                    }
                    for(i=0;i<qtdEntradas;i++){
                         redeNeural.gradienteCamada1[i] = (dados[k].output-redeNeural.saidaCamada1[i])*redeNeural.saidaCamada1[i];
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada1[i],redeNeural.saidaCamada1[i],1);
                    }
                    for(i=0;i<qtdEntradas;i++){
                         redeNeural.gradienteCamada0[i] = (dados[k].output-redeNeural.saidaCamada0[i])*redeNeural.saidaCamada0[i];
                         redeNeural.recalcularPesos(redeNeural.gradienteCamada0[i],redeNeural.saidaCamada0[i],0);
                    }
               }
               eqmAtual = redeNeural.calcularEqm(dados,2);
               console.log("eqmAtual: ");
               console.log(eqmAtual);
               numEpoca++;
          }
          redeNeural.showEpocas = numEpoca;
          console.log("Epocas: "+numEpoca);
          console.log("Pesos Finais:");
          for(z=0;z<3;z++){
               console.log("wf"+z+": "+redeNeural.pesosCamada[z]);
          }
     }
}

