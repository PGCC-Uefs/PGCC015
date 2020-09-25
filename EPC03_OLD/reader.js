// Mestrado PGCC Uefs 2020.1
// Perceptron Multicamadas EPC03
// objeto de leitura dos arquivos .dat e execução do treinamento
// Noberto Pires Maciel

this.ler = function(){
        var leitura = this;
        // this.dados = [];
        this.fileSelected = document.getElementById('txtfiletoread');
        this.fileExtension = /.*/; //application/octet-stream //zz-application/zz-winassoc-dat
        this.fileTobeRead = leitura.fileSelected.files[0];
        this.executa = function(){
            if(leitura.fileTobeRead != undefined){
                console.log("Mime type: "+leitura.fileTobeRead.type);        
                if (leitura.fileTobeRead.type.match(leitura.fileExtension)) {
                    var fileReader = new FileReader();
                    fileReader.onload = function () {
                        var array = fileReader.result.split("\r\n");
                        var amostras = [];
                        var y = array.indexOf("@data")+1;
                        var x = 0;
                        while(y<array.length){
                                let arraySlipt = array[y].split(",");
                                let output = arraySlipt.pop();
                                let inputs = arraySlipt;
                                amostras[x] =   {
                                                    inputs: inputs, 
                                                    output: output
                                                };
                            y++;
                            x++;
                        }
                        if(amostras[parseInt(x-1)].inputs.length == 0){amostras.pop();}
                        this.dados = amostras;
                        // var neuronio = new pmc();
                        // neuronio.treinamento(amostras);
                    }
                    fileReader.readAsText(leitura.fileTobeRead);
                }
                else {
                    alert("Por favor selecione arquivo texto");
                }
            }
            else {
                alert("Por favor selecione um arquivo");
            }
        }
}