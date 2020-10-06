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
     this.numEpocas = 10**4;
     this.erro = [];
     this.topologia = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.e = 2.718281828;
     this.alfa = 0.5; // alfa do termo de momentum
     this.sinalGradiente = 1;
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
          //console.log("Pesos iniciais:");
          var camadas = redeNeural.numCamadas;
          var pesosPorCamada = [];
          for(k=0;k<camadas;k++){
               var neuronios = [];
               var topologia = redeNeural.topologia[k];
               for(j=0;j<topologia;j++){
                    if(k == 0 || k == redeNeural.topologia[redeNeural.topologia.length-1]){entradas = qtdEntradas}
                    else{entradas = topologia+1}
                    var pesos = [];
                    for(i=0;i<entradas;i++){
                         pesos[i] = Math.random()/2;
                         console.log("T:"+(topologia)+" C:"+k+" N:"+j+" wi"+i+": "+pesos[i]);
                         pesosPorCamada[k] = i*j;
                    }
                    neuronios[j] = {pesos: pesos};
               }
               redeNeural.camada[k] = {neuronios: neuronios, wbefore: neuronios};
          }
          //console.log("pesos por camada: "+JSON.stringify(pesosPorCamada));
     }
     this.calcularU = function(dados,n){
          var camada = redeNeural.camada[n].neuronios;
          var qtdNeuronios = camada.length;
          var arraySaida = [];
          var apresenta = "";
          if(n>0){dados = redeNeural.camada[n-1].outputs; apresenta = redeNeural.camada[n-1].outputs};
          //console.log("calcularU, entrada camada: "+n+"\noutputs["+(n-1)+"]: "+JSON.stringify(apresenta[0])+"\nqtdNeuronios: "+qtdNeuronios+"\ndadosNeuronios: "+JSON.stringify(redeNeural.camada[n]));
          for(z=0;z<dados.length;z++){
               for(i=0;i<qtdNeuronios;i++){
                    var soma = 0;
                    var pesos = camada[n].pesos;
                    //console.log("calcularU laço 1: \nn="+n+"\nneuronio: "+i+"\npesos: "+JSON.stringify(pesos));
                    for(j=0;j<pesos.length;j++){
                         soma += dados[z] * pesos[j];
                         //console.log("U laço 2: \nC="+n+"\nN:"+i+"\nI["+z+"]:"+dados[z]+"\nP["+j+"]:"+pesos[j]+"\nS:"+soma);
                                             //console.log("pesos: "+JSON.stringify(pesos));
                    }
                    (n==redeNeural.camada.length-1? arraySaida[i] = soma : arraySaida[i] = soma + redeNeural.bias);
               }
          }
          redeNeural.camada[n].outputsU = arraySaida;
          //console.log("calcularU saída camada: n="+n+"\noutputsU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\ndados: "+JSON.stringify(dados));
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
          //console.log(JSON.stringify("Funçao ativação camada: "+n+"\noutputs: "+redeNeural.camada[n].outputs)+"\nNeuronios: "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.dSigmoid = function(n){
          var outputs = redeNeural.camada[n].outputs;
          var dSigmoid = [];
          var k = redeNeural.k;               // declividade
          var l = redeNeural.l;               // valor máximo da curva
          var e = redeNeural.e;              // número neperiano
          for(i=0;i<outputs.length;i++){
               dSigmoid[i] = k*outputs[i]*(1-outputs[i]); //k*(e**-k*outputs[i])/(((e**-k*outputs[i])+1)**2); //(e**-outputs[i])*(l+(e**-outputs[i]))**2; //outputs[i]*(1-outputs[i]); //outputs[i]*(1-outputs[i]);
          }
          redeNeural.camada[n].dSigmoid = dSigmoid;
          //console.log("dSigmoid pesos camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.calcularGradiente = function(k,n){
          var gradiente = [];
          var sinal = redeNeural.sinalGradiente;
          var outputsNeuronios = []; // transformado em array
          var camada = redeNeural.camada[n];
          (n == redeNeural.camada.length-1? outputsNeuronios = redeNeural.amostras[k].output.split("") : outputsNeuronios = camada.outputs);
          alert(camada.neuronios.length+" - "+n);
          for(i=0;i<camada.neuronios.length;i++){
               var dK = outputsNeuronios[i];
               var output = camada.outputs[i];
               var soma = 0;
               
               //
               // GRADIENTE ESTÁ ZERANDO
               // ERRO AQUI: (dK-output)
               //gradiente[i] = sinal*(dK-output)*dSigmoid;//sinal*(output-camada.dSigmoid[i]);]
               if(n == redeNeural.camada.length-1){ // se for == ultima camada
                    //var outputY = redeNeural.camada[n-1].outputs;
                    //gradiente[i] = sinal*(dK-output)*output*(1-output)*outputY[i];
                    //alert(JSON.stringify(outputY));
                    var dSigmoid = redeNeural.camada[n-1].dSigmoid[i];
                    gradiente[i] = sinal*(dK-output)*dSigmoid;
               }
               else{
                    var c = redeNeural.camada[n+1];
                    for(j=0;j<c.neuronios.length;j++){
                         var d = camada.dSigmoid[j];
                         var g = c.gradiente[j];
                         var w = c.neuronios[j].pesos;
                         alert(w);
                         soma += g*w*d;
                    }
                    gradiente[i] = soma;
               }
               //
               // -(y1-go1)*go1(1-go1)

               //console.log("Gradiente["+i+"]: camada ["+n+"]\nvalor: "+gradiente[i]+"\ndK["+i+"]: "+dK+"\noutput["+i+"]"+output+"\ndSigmoid["+i+"]"+dSigmoid);     
          }
          redeNeural.camada[n].gradiente = gradiente;
          //console.log("Gradientes camada["+n+"]: "+JSON.stringify(redeNeural.camada[n].gradiente));
     }
     this.recalcularPesos = function(k,n){
          var camada = redeNeural.camada[n];
          var inputs = redeNeural.amostras[k].inputs
          var qtdNeuronios = camada.neuronios.length;
          //console.log("recalcularPesos inicio laço 1 camada n="+n+"\nqtdNeuronios: "+qtdNeuronios+"\ninputs"+JSON.stringify(inputs));
          for(i=0;i<qtdNeuronios;i++){
               var saida = [];
               var wbefore = camada.wbefore[i].pesos;
               var pesos = camada.neuronios[i].pesos;
               (n>0? saida = redeNeural.camada[n-1].outputs : saida = inputs);
               //alert("Saída como entrada para o ngy: "+JSON.stringify(saida));
               //console.log("recalcularPesos inicio laço 2 camada["+n+"], "+qtdNeuronios+"neuronios. \npesos"+JSON.stringify(pesos));
               for(j=0;j<pesos.length;j++){
                    //console.log("recalcularPesos INTERNO laço 2 camada["+n+"], "+qtdNeuronios+"neuronios. \nsaida["+i+"]: "+saida[i]);
                    var alfa = redeNeural.alfa;
                    var ngy = pesos[j] + redeNeural.txAprendizagem * camada.gradiente[i] * saida[i];
                    //console.log("recalcularPesos INTERNO laço 2 camada["+n+"], "+qtdNeuronios+"neuronios.\ngradiente["+i+"]: "+camada.gradiente[i]);
                    
                    var momentum = alfa*(ngy - wbefore[j]);
                    
                    // o termo de momentum leva em conta o w da época

                    //console.log("recalcularPesos INTERNO laço 2 camada["+n+"], "+qtdNeuronios+"neuronios.\nwbefore["+j+"]: "+wbefore[j]);
                    
/*                     alert("TESTE DE MESA"+
                    "\nj (pesos.length): "+j+
                    "\npesos length: "+pesos.length+
                    "\nk (amostra): "+k+
                    "\nn (camada): "+n+
                    "\ngradiente: "+camada.gradiente[i]+
                    "\naprendizagem: "+redeNeural.txAprendizagem+
                    "\nqtdNeuronios: "+qtdNeuronios+
                    "\nInputs: "+inputs.length+
                    "\npesos: "+JSON.stringify(pesos)+
                    "\nSaida length: "+saida.length+
                    "\nSaida: "+saida[i]+
                    "\nngy: "+ngy+
                    "\nwBefore: "+wbefore[j]+
                    "\nmomentum: "+momentum+
                    "\npesos: "+JSON.stringify(pesos[j])); */

                    pesos[j] = ngy + momentum;
               }
               camada.neuronios[i].pesos = pesos;
               //console.log("Pesos gerais\ncamada: "+n+"\nneuronio: "+i+"\n"+JSON.stringify(camada));
          }
          //console.log("wBefore: "+n+" -> "+JSON.stringify(camada.wbefore));
     }
     this.erroAmostra = function(dK){
          var erroAmostra = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = dK.length; //redeNeural.camada[n].outputs.length;
          for(i=0;i<qtd;i++){
               soma += dK[i]-redeNeural.camada[n].outputs[i]; // falha aqui, outputs(len=4) superando dK(len=4)
               //alert("erroAmostra camada: "+n+"\nsoma: "+soma+"\nqtdOutputs: "+qtd+"\noutputs: "+redeNeural.camada[n].outputs[i]+"\ndK["+i+"]: "+dK[i]);
          }
          erroAmostra = (soma**2)/2;
          //alert("erroAmostra: "+erroAmostra+"\nsoma: "+soma+"\ndK: "+JSON.stringify(dK));
          return erroAmostra;
     }
     this.calcularEqm = function(){
          var soma = 0;
          var eqm = 0;
          var p = redeNeural.amostras.length;
          for(i=0;i<p;i++){
               soma += redeNeural.erro[i];
          }
          eqm =  soma/p;
          //console.log("calcularEqm EQM: "+eqm+"\np: "+p+"\nsoma: "+soma);
          return eqm;
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 999999999999999999999999999999999;
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
                         //console.log("camada laço treinamento: "+n+"\ninput: "+input);
                         redeNeural.calcularU(input,n);
                         redeNeural.funcaoAtivacao(n);
                         redeNeural.dSigmoid(n);
                         input = redeNeural.camada[n].outputs;
                         if(n==0 || n<(camadas-1)){input.unshift(redeNeural.bias);} // acrescenta bias na saída da camada 2 em diante, menos na última camada
                         //alert("Treinamento \ncamada: "+n+"\nneuronio: "+i+"\n"+JSON.stringify(redeNeural.camada[n].neuronios));
                    }
                    for(n=(camadas-1);n>=0;n--){
                         redeNeural.calcularGradiente(k,n);
                         redeNeural.recalcularPesos(k,n);
                    }
                    redeNeural.erro[k] = redeNeural.erroAmostra(dados[k].output);
               }
               eqmAtual = redeNeural.calcularEqm();
               //alert(Math.abs(eqmAtual - eqmAnterior));
               for(n=0;n<camadas;n++){
                    let camada = redeNeural.camada[n];
                    for(i=0;i<camada.length;i++){
                         camada.wbefore[i].pesos = redeNeural.camada[n].neuronios[i].pesos;
                    }
               }
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
          console.log("Amostras: "+(k));
          console.log("Epocas: "+numEpoca);
     }
     this.executar = function(dados){
          var numCamadas = redeNeural.numCamadas;
          var resultado = [];
          for(k=0;k<dados.length;k++){
               var input = dados[k].inputs;
               //console.log(JSON.stringify(input));
               for(n=0;n<numCamadas;n++){
                    redeNeural.calcularU(input,n);
                    redeNeural.funcaoAtivacao(n);
                    //console.log("k="+k+"\ncamada["+n+"]\nnum camadas: "+numCamadas+"\ninputs teste: "+JSON.stringify(input)+"\noutputs teste: "+redeNeural.camada[n].outputs+"\ntamanho de dados: "+dados.length);
                    input = redeNeural.camada[n].outputs;
                    //if(n>0 || n<(numCamadas-1)){input.unshift(redeNeural.bias);}
                    //console.log(JSON.stringify(redeNeural.camada[numCamadas-1].outputsU));
               }
               resultado[k] = redeNeural.camada[numCamadas-1].outputs;
               //console.log(JSON.stringify(input));
               console.log(JSON.stringify(resultado[k]));
          }
     }

}

