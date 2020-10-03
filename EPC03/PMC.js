// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// Noberto Pires Maciel

console.log("PMC:");

this.pmc = function(){
     var redeNeural = this;
     this.bias = -1;
     this.amostras = []; 
     this.numEpocas = 10**6;
     this.eq = [];
     this.topologia = [];
     this.txAprendizagem = 0.1;
     this.precisao = 10**-6;
     this.e = 2.718281828;
     this.alfa = 0.5; // alfa do termo de momentum
     this.sinalGradiente = -1;
     this.camada = [];
     this.wbefore = [];
     this.numCamadas = [];
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
          console.log("Pesos iniciais:");
          var camadas = redeNeural.numCamadas;
          for(k=0;k<camadas;k++){
               var neuronios = [];
               var wbefore = [];
               var topologia = redeNeural.topologia[k];
               for(j=0;j<topologia;j++){
                    (k > 0? entradas = redeNeural.topologia[k-1] : entradas = qtdEntradas);
                    var pesos = [];
                    for(i=0;i<entradas;i++){
                         pesos[i] = Math.random()/2;
                         console.log("Camada: "+k+" wi"+i+": "+pesos[i]);
                    }
                    neuronios[j] = {pesos: pesos};
                    wbefore[j] = neuronios[j];
               }
               redeNeural.camada[k] = {neuronios: neuronios, wbefore: wbefore};
          }
     }
     this.calcularU = function(dados,n){
          var camada = redeNeural.camada[n].neuronios;
          var arraySaida = [];
          (n>0? dados = redeNeural.camada[n-1].outputs : dados);
          for(i=0;i<camada.length;i++){
               var soma = 0;
               var pesos = camada[i].pesos;
               for(j=0;j<pesos.length;j++){
                    soma += dados[j] * pesos[j];
               }
               (redeNeural.camada.length < n? arraySaida[i] = soma + redeNeural.bias : arraySaida[i] = soma);
          }
          redeNeural.camada[n].outputsU = arraySaida;
          //alert("Camada: "+n+"\nneurônio: "+i+"\npesos: "+pesos.length+"\ninputs: "+dados.length);
          //document.getElementById("dadosSaida").append("\ncamada: "+n+"\nU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\n\n");
          //alert("\ncamada: "+n+"\nU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\n\n");
     }
     this.funcaoAtivacao = function(n){
          var camada = redeNeural.camada[n];
          var ativacao = [];
          var bias = redeNeural.bias;
          var k = 1;               // declividade
          var x0 = 0;              // ponto médio da curva sigmoide
          var l = 1;               // valor máximo da curva
          var e = redeNeural.e;    // número neperiano
          for(i=0;i<camada.outputsU.length;i++){
               ativacao[i] = l/(1 + e*Math.exp(-k*(camada.outputsU[i]-x0)));
          }
          (n<redeNeural.camada.length-1)?ativacao.unshift(bias):0;
          redeNeural.camada[n].outputs = ativacao;
          //document.getElementById("dadosSaida").append("\ncamada: "+n+"\nATIVAÇÃO: "+JSON.stringify(redeNeural.camada[n])+"\n\n");
          //alert("\ncamada: "+n+"\nativação: "+JSON.stringify(redeNeural.camada[n].outputs)+"\n\n");
     }
     this.derivada = function(n){
          var outputs = redeNeural.camada[n].outputs;
          var derivada = [];
          var l = 1;               // valor máximo da curva
          var e = redeNeural.e;
          for(i=0;i<outputs.length;i++){
               derivada[i] = (e**-outputs[i])*(l+(e**-outputs[i]))**2;
          }
          redeNeural.camada[n].derivada = derivada;
     }
     this.calcularGradiente = function(k,n){
          var gradiente = [];
          var sinal = redeNeural.sinalGradiente;
          var outputAmostra = redeNeural.amostras[k].output;
          var camada = redeNeural.camada[n];
          var camadaAnterior = 0;
          (n > 0 ? camadaAnterior = redeNeural.camada[n-1] : camadaAnterior = 0);
          (n> 0? outputLen = camada.outputs.length : outputLen = redeNeural.amostras[k].inputs.length);
          for(i=0;i<outputLen;i++){
               (n>0 ? gradiente[i] = sinal*(outputAmostra-camada.outputs[i])*camadaAnterior.derivada[i] : gradiente[i] = sinal*(outputAmostra-camada.outputs[i])*redeNeural.amostras[k].inputs[i]); //gradiente saida 3 = -(dj³-Yj³)*g'(Ij³), onde I³ = Y²
               //alert("camada: "+n+"\noutputs length: "+camada.outputs.length+"\ni: "+i+"\ngradiente: "+gradiente[i]+"\noutput amostra k: "+outputAmostra+"\noutput: "+camada.outputs[i]+"\ninput[i] amostra k: "+redeNeural.amostras[k].inputs[i]);
          }
          redeNeural.camada[n].gradiente = gradiente;
          //document.getElementById("dadosSaida").append("\ncamada: "+n+"\nGRADIENTE: "+JSON.stringify(redeNeural.camada[n])+"\n\n");
          //alert("\ncamada: "+n+"\ngradiente: "+JSON.stringify(redeNeural.camada[n])+"\n\n");
     }
     this.recalcularPesos = function(k,n){
          var camada = redeNeural.camada[n];
          var inputs = redeNeural.amostras[k].inputs
          for(i=0;i<camada.neuronios.length;i++){
               var saida = [];
               var wbefore = camada.wbefore[i].pesos;
               var pesos = camada.neuronios[i].pesos;
               camada.wbefore[i] = {pesos: pesos};
               (n>0? saida = redeNeural.camada[n-1].outputs : saida = inputs);
               for(j=0;j<pesos.length;j++){
                    let alfa = redeNeural.alfa;
                    let ngy = pesos[j] + redeNeural.txAprendizagem * camada.gradiente[i] * saida[i];
                    let momentum = alfa*(ngy - wbefore[j]); // ngy = txAprendizado*gradeiante*saída Y
                    pesos[j] = ngy + momentum;
               }
               camada.neuronios[i] = {pesos: pesos};
          }
          //document.getElementById("dadosSaida").append("\ncamada "+n+"\nqtdSaidas: "+saida.length+"\nRECALCULAR Pesos: "+JSON.stringify(camada.neuronios)+"\n\n");
          //alert("\ncamada "+n+"\nqtdSaidas: "+saida.length+"\nrecalcularPesos: "+JSON.stringify(camada.neuronios).replace(/{/g, '\n')+"\n\n");
     }
     this.eqAmostra = function(dK){
          var eqAmostra = 0;
          var soma = 0;
          var n = (redeNeural.camada.length-1);
          var qtd = redeNeural.camada[n].outputs.length;
          for(i=0;i<qtd;i++){
               soma += dK-redeNeural.camada[n].outputs[i];
          }
          eqAmostra = (soma**2)/2;
          return eqAmostra;
     }
     this.calcularEqm = function(){
          var soma = 0;
          var eqm = 0;
          var p = redeNeural.amostras.length;
          for(i=0;i<p;i++){
               soma += redeNeural.eq[i];
          }
          eqm =  soma/p;
          //alert(eqm);
          return eqm;
     }
     this.treinamento = function(dados){
          var numEpoca = 0;
          var eqmAnterior = 999999999999999999999999;
          var eqmAtual = 0;
          var camadas = redeNeural.numCamadas;
          var qtdEntradas = dados[0].inputs.length;
          redeNeural.amostras = dados;
          redeNeural.iniciaPesos(qtdEntradas);
          while((Math.abs(eqmAtual - eqmAnterior) > redeNeural.precisao)){
               var erro = Math.abs(eqmAtual - eqmAnterior);
               eqmAnterior = eqmAtual;
               for(k=0;k<redeNeural.amostras.length ;k++){
                    var input = dados[k].inputs;
                    for(n=0;n<camadas;n++){
                         redeNeural.calcularU(input,n);
                         redeNeural.funcaoAtivacao(n);
                         redeNeural.derivada(n);
                         input = redeNeural.camada[n].outputs;
                    }
                    redeNeural.eq[k] = redeNeural.eqAmostra(dados[k].output);
                    for(m=(camadas-1);m>=0;m--){
                         redeNeural.calcularGradiente(k,m);
                         redeNeural.recalcularPesos(k,m);
                    }
               }
               eqmAtual = redeNeural.calcularEqm();
               numEpoca++;
               //alert("Época: "+numEpoca+"\nEqmAtual: "+eqmAtual+"\nErro: "+erro);
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
          console.log("Amostras: "+(k-1));
          console.log("Epocas: "+numEpoca);
     }
     this.executar = function(dados){
          var camadas = redeNeural.numCamadas;
          var resultado = [];
          for(k=0;k<dados.length;k++){
               var inputs = dados[k].inputs;
               for(n=0;n<camadas;n++){
                    redeNeural.calcularU(inputs,n);
                    alert("\ninputs: "+inputs[0]+"\ncamada: "+n+"\nU: "+JSON.stringify(redeNeural.camada[n].outputsU)+"\n\n");
                    redeNeural.funcaoAtivacao(n);
               }
               resultado[k] = redeNeural.camada[camadas-1].outputs;
          }
          alert(JSON.stringify(resultado));
          return resultado;
     }

}

