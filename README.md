# Hand Gesture Recognition

Demonstração Java/OpenCV que detecta contornos candidatos a mãos por segmentação de cor. Ainda não classifica gestos e pode falhar com iluminação, fundo e diferentes tons de pele.

## Como usar
Instale JDK e OpenCV Java (JAR e biblioteca nativa da mesma versão e arquitetura). Adicione o JAR ao classpath. Configure `OPENCV_JAVA_LIBRARY` com o caminho absoluto da biblioteca nativa, ou use `-Djava.library.path` com a pasta dela. Compile `javac -cp CAMINHO_DO_JAR -d out src/HandGestureRecognition.java` e execute `java -cp "out:CAMINHO_DO_JAR" HandGestureRecognition` (no Windows, use ponto e vírgula em vez de dois pontos). Pressione Esc para sair.

A câmera é processada localmente. Teste com a instalação de OpenCV do seu sistema.
