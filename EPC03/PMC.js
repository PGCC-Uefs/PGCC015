// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.beta = 0.5;               // declividade
     this.x0 = 0;              // ponto médio da curva sigmoide
     this.l = 1;
     this.amostras = []; 
     this.numEpocas = 10**6;
     this.erro = [];
     this.topologia = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.e = 2.718281828459045;
     this.alfa = 0.1; // alfa do termo de momentum
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
          var camadas = redeNeural.numCamadas;
          var pesosPorCamada = [];
          for(k=0;k<camadas;k++){
               var neuronios = [];
               var topologia = redeNeural.topologia[k];
               var entradas = qtdEntradas;
               if(k > 0){entradas = redeNeural.topologia[k-1]+1;}
               for(j=0;j<topologia;j++){
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
     }
     this.calcularU = function(inputs,n){
          var neuronio = redeNeural.camada[n].neuronios;
          var qtdNeuronios = neuronio.length;
          var arraySaida = [];
          var tam = inputs.length;
          for(i=0;i<qtdNeuronios;i++){
                    var soma = 0;
                    var y = 0;
                    var pesos = neuronio[i].pesos;
                    for(j=0;j<tam;j++){
                         soma += inputs[j] * pesos[j+y];
                         //alert("Camada: "+n+"\nNeurônio: "+i+"\ninputs["+j+"]: "+inputs[j]+"\npesos["+(y+j)+"]: "+pesos[y+j]);
                    }
                    y += j;
                    arraySaida[i] = soma; //+ redeNeural.bias;
          }
          redeNeural.camada[n].outputsU = arraySaida;
          //alert(JSON.stringify(redeNeural.camada[n].outputsU));
          //console.log("calcularU saída camada: n="+n+"\noutputsU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\ndados: "+JSON.stringify(inputs));
     }
     this.funcaoAtivacao = function(n){
          var neuronios = redeNeural.camada[n];
          var ativacao = [];
          var beta = redeNeural.beta;               // declividade
          var x0 = redeNeural.x0;              // ponto médio da curva sigmoide
          var l = redeNeural.l;               // valor máximo da curva
          var e = redeNeural.e;    // número neperiano
          for(i=0;i<neuronios.outputsU.length;i++){
               ativacao[i] = l/(1 + e**(-beta*(neuronios.outputsU[i]-x0)));
               //ativacao[i] = l/(1 + Math.exp(-beta*(neuronios.outputsU[i]-x0)));
          }
          redeNeural.camada[n].outputs = ativacao;
     }
     this.derivadaSigmoid = function(n){
          var outputs = redeNeural.camada[n].outputs;
          var derivadaSigmoid = [];
          var beta = redeNeural.beta;               // declividade
          var l = redeNeural.l;               // valor máximo da curva
          var e = redeNeural.e;              // número neperiano
          for(i=0;i<outputs.length;i++){
               derivadaSigmoid[i] = beta*outputs[i]*(1-outputs[i]); //k*(e**-k*outputs[i])/(((e**-k*outputs[i])+1)**2); //(e**-outputs[i])*(l+(e**-outputs[i]))**2; //outputs[i]*(1-outputs[i]); //outputs[i]*(1-outputs[i]);
          }
          redeNeural.camada[n].derivadaSigmoid = derivadaSigmoid;
          //console.log("derivadaSigmoid pesos camada: "+n+" -> "+JSON.stringify(redeNeural.camada[n].neuronios));
     }
     this.calcularGradiente = function(k,n){
          var gradiente = [];
          var sinal = redeNeural.sinalGradiente;
          var camada = redeNeural.camada[n];
          var lastY = redeNeural.amostras[k].output.split("");
          for(i=0;i<camada.neuronios.length;i++){
               var soma = 0;
               if(n == redeNeural.camada.length-1){
                    var output = camada.outputs[i];
                    var derivadaSigmoid = redeNeural.camada[n-1].derivadaSigmoid[i];
                    gradiente[i] = sinal*(lastY[i]-output)*derivadaSigmoid;
/*                     console.log("Gradiente SAÍDA SE n == ultima camada:"+
                    "\nlastY["+i+"]: "+lastY[i]+
                    "\noutput: "+output+
                    "\nderivadaSigmoide: "+derivadaSigmoid+
                    "\ngradiente: "+gradiente[i]+
                    "\ncamada[n]: "+n+
                    "\nneuronio[i]: "+i); */
               }
               else{
                    var c = redeNeural.camada[n+1];
                    var d = camada.derivadaSigmoid[i];
                    for(j=0;j<c.neuronios.length;j++){
                         var g = c.gradiente[j];
                         var w = c.neuronios[j].pesos;
                         for(p=0;p<w.length;p++){
                              soma += g*w[p];
                         }
                         // erro aqui, sigmoid undefined n+1 bate key errada
/*                          console.log("Gradiente SAÍDA SE n < ultima camada:"+
                         "\nneuronio[j]: "+j+
                         "\ncamada[n]: "+n+
                         "\noutput: "+output+
                         "\npesos: "+w+
                         "\nderivadaSigmoide: "+d+
                         "\ngradiente: "+g+
                         "\nsoma[j]: "+soma); */
                    }
                    gradiente[i] = soma*d;
               }
          }
          redeNeural.camada[n].gradiente = gradiente;
          //console.log("Gradientes camada["+n+"]: "+JSON.stringify(redeNeural.camada[n].gradiente));
     }
     this.recalcularPesos = function(k,n){
          var camada = redeNeural.camada[n];
          var inputs = redeNeural.amostras[k].inputs
          var qtdNeuronios = camada.neuronios.length;
          //alert("recalcularPesos inicio laço 1 camada n="+n+"\nqtdNeuronios: "+qtdNeuronios+"\ninputs"+JSON.stringify(inputs));
          //console.clear();
          for(i=0;i<qtdNeuronios;i++){
               var saida = [];
               var wbefore = camada.wbefore[i].pesos;
               var pesos = camada.neuronios[i].pesos;
               (n>0? saida = redeNeural.camada[n-1].outputs : saida = inputs); // erro está no numero de saídas CORRIGIR
               for(j=0;j<saida.length;j++){
                    var alfa = redeNeural.alfa;
                    var ngy = pesos[j] + redeNeural.txAprendizagem * camada.gradiente[i] * saida[j];
                    var momentum = alfa*(ngy - wbefore[j]);                    
/*                     console.log("RECÁLCULO DOS PESOS:"+
                    "\nn (camada): "+n+
                    "\nneurônio: "+i+
                    "\nk (amostra): "+k+
                    "\nj (pesos.length): "+j+
                    "\npesos length: "+pesos.length+
                    "\ngradiente["+i+"]: "+camada.gradiente[i]+
                    "\naprendizagem: "+redeNeural.txAprendizagem+
                    "\nerro de k["+k+"]: "+redeNeural.erro[k]+
                    "\nqtdNeuronios: "+qtdNeuronios+
                    "\nInputs: "+inputs.length+
                    "\nSaida length: "+saida.length+
                    "\nSaida["+j+"]: "+saida[j]+
                    "\nngy: "+ngy+
                    "\nwBefore: "+wbefore[j]+
                    "\nmomentum: "+momentum+
                    "\npesos: "+JSON.stringify(pesos)+
                    "\npesos["+j+"]: "+JSON.stringify(pesos[j])); */

                    pesos[j] = ngy + momentum;
               }
               camada.neuronios[i].pesos = pesos;
          }
     }
     this.erroAmostra = function(dK){
          var erroAmostra = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = dK.length; //redeNeural.camada[n].outputs.length;
          for(i=0;i<qtd;i++){
               soma += dK[i]-redeNeural.camada[n].outputs[i]; // falha aqui, outputs(len=4) superando dK(len=4)
               //soma = soma**2;
               //alert("erroAmostra camada: "+n+"\nsoma: "+soma+"\nqtdOutputs: "+qtd+"\noutputs: "+redeNeural.camada[n].outputs[i]+"\ndK["+i+"]: "+dK[i]);
          }
          erroAmostra = (soma**2)/2;
          //erroAmostra = (soma)/2;
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
          //alert("calcularEqm EQM: "+eqm+"\np: "+p+"\nsoma: "+soma);
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
                         redeNeural.calcularU(input,n);
                         redeNeural.funcaoAtivacao(n);
                         //alert("Saída: "+JSON.stringify(redeNeural.camada[camadas-1].outputs)+"\ndados: "+JSON.stringify(dados[k].output));
                         redeNeural.derivadaSigmoid(n);
                         input = redeNeural.camada[n].outputs;
                         if(n>0 && n<camadas-1){input.unshift(redeNeural.bias);} // acrescenta bias na saída da camada 2 em diante, menos na última camada
                    }
                    for(n=(camadas-1);n>=0;n--){
                         redeNeural.calcularGradiente(k,n);
                         redeNeural.recalcularPesos(k,n);
                    }
                    redeNeural.erro[k] = redeNeural.erroAmostra(dados[k].output);
               }
               eqmAtual = redeNeural.calcularEqm();
               for(n=0;n<camadas;n++){
                    for(i=0;i<redeNeural.camada[n].length;i++){
                         redeNeural.camada[n].wbefore[i].pesos = redeNeural.camada[n].neuronios[i].pesos;
                    }
               }
               numEpoca++;
               //console.log("Treinamento epoca:"+numEpoca+"\nneuronios: "+JSON.stringify(redeNeural.camada));
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
                    //alert("Input["+n+"]: "+JSON.stringify(input)+"\nSaída de U["+n+"]: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\nSaída de sigmoid["+n+"]: "+JSON.stringify(redeNeural.camada[n].outputs));
                    //console.log("k="+k+"\ncamada["+n+"]\nnum camadas: "+numCamadas+"\ninputs teste: "+JSON.stringify(input)+"\noutputs teste: "+redeNeural.camada[n].outputs+"\ntamanho de dados: "+dados.length);
                    if(n<numCamadas-1){input = redeNeural.camada[n].outputs;}
                    //if(n>0 || n<(numCamadas-1)){input.unshift(redeNeural.bias);}
                    //console.log(JSON.stringify(redeNeural.camada[numCamadas-1].outputsU));
               }
               resultado[k] = redeNeural.camada[numCamadas-1].outputs;
               console.log(JSON.stringify(resultado[k]));
          }
     }

}

