this.perceptron = function(){
    var redeNeural = this;
    this.inicializaPesos = function(entradas,pesos){
        for(i=0;i<entradas;i++){
            redeNeural.pesos[i] = parseInt(Math.random()*10);
        }
    }
    this.taxa = 0.01;
    this.bias = dados[0].inputs;
    this.pesos = [];
    var erro = true;
    //Inicializar o vetor de pesos com valores aleatórios;
    //Inicializar a taxa de aprendizado;
    while(erro){
        //Repita
        // Erro “não existe”;
        // Para cada par de treinamento {x(k),d(k)} faça
        for(i=0;i<dados.length;i++){

        }
    //u <- x(k)
    //T . w;
    //y <- Sinal(u);
    //Se (d(k) ≠ y) então
    //wi <- wi-1 + .(d(k) - y).x(k)
    //Erro <- “existe”;
    //fim_se;
    //fim_para;
    //Até Erro = “não existe”;
    }

  
}