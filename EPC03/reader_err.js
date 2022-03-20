// Retirado de: https://codigosimples.net/2016/04/25/ler-um-arquivo-local-usando-html5-e-javascript/
// Remodelagem e ajustes para leitura dos arquivos .DAT feita por Noberto Maciel

this.ler = function(){
        var leitura = this;
        this.dados = [];
        this.lerArquivo = function(){
                var fileSelected = document.getElementById('txtfiletoread');
                var mimeType = /.*/; //application/octet-stream //zz-application/zz-winassoc-dat
                var fileTobeRead = fileSelected.files[0];
                console.log("Mime type: "+fileTobeRead.type);
                console.log(fileTobeRead);
                if (fileTobeRead.type.match(mimeType)) {
                    fileReader = new FileReader();
                    fileReader.carregar = function () {
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
                        console.log(JSON.stringify(amostras));
                        leitura.dados = amostras;
                    }
                    fileReader.readAsText(fileTobeRead);
                }
                else {
                    alert("Por favor selecione arquivo texto");
                }
        }
        return leitura.lerArquivo();
}