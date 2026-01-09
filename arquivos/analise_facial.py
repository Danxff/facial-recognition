import cv2
import numpy as np
from deepface import DeepFace

nome_arquivo_video = 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
arquivo_relatorio = 'relatorio_projeto.txt'

SENSIBILIDADE_MOVIMENTO = 10000 

def processar_projeto_completo():
    cap = cv2.VideoCapture(nome_arquivo_video)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    total_frames = 0
    total_anomalias = 0
    contagem_emocoes = {} 
    
    # Variáveis de Controle
    frame_anterior = None
    faces_detectadas = [] # Cache para não rodar IA em todo frame
    texto_anomalia = ""
    cor_anomalia = (0, 255, 0) # Verde (normal)

    print(f"Iniciando análise completa em: {nome_arquivo_video}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        
        # 1. Preparação para Detecção de Movimento
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0) # Borrar remove ruídos (chuviscos)

        # Se for o primeiro frame, apenas guardamos e pulamos
        if frame_anterior is None:
            frame_anterior = frame_gray
            continue

        # 2. Cálculo de Diferença (Movimento)
        # Calcula a diferença absoluta entre o frame atual e o anterior
        diferenca = cv2.absdiff(frame_anterior, frame_gray)
        # O que for diferente (acima de 25) vira branco, o resto preto
        _, thresh = cv2.threshold(diferenca, 25, 255, cv2.THRESH_BINARY)
        # Conta quantos pixels brancos (mudanças) existem
        contagem_pixels_mudaram = cv2.countNonZero(thresh)

        # Atualiza o frame anterior para a próxima rodada
        frame_anterior = frame_gray

        # 3. Classificação de Anomalia
        if contagem_pixels_mudaram > SENSIBILIDADE_MOVIMENTO:
            texto_anomalia = "ALERTA: Movimento Brusco Detectado!"
            cor_anomalia = (0, 0, 255) # Vermelho
            total_anomalias += 1
        else:
            texto_anomalia = "Status: Normal"
            cor_anomalia = (0, 255, 0) # Verde

        # 4. Reconhecimento Facial e Emoções (A cada 10 frames para otimizar)
        if total_frames % 10 == 0:
            try:
                analise = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                faces_detectadas = analise
                
                # Contabilizar emoções para o relatório
                if isinstance(analise, list):
                    for rosto in analise:
                        emo = rosto['dominant_emotion']
                        # Se a emoção já existe no dicionário, soma 1. Se não, começa com 1.
                        contagem_emocoes[emo] = contagem_emocoes.get(emo, 0) + 1
            except:
                faces_detectadas = []

        # 5. Desenhar na Tela (Output Visual)
        # Texto de Anomalia/Status no topo
        cv2.putText(frame, f"{texto_anomalia} (Intensidade: {contagem_pixels_mudaram})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_anomalia, 2)

        # Retângulos nos rostos
        if isinstance(faces_detectadas, list):
            for rosto in faces_detectadas:
                region = rosto['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                emocao = rosto['dominant_emotion']
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Sistema de Monitoramento - Pós Tech', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Pós-Processamento: Gerar Relatório ---
    gerar_relatorio(total_frames, total_anomalias, contagem_emocoes)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Processamento concluído. Relatório gerado.")

def gerar_relatorio(frames, anomalias, emocoes):
    """Cria um arquivo de texto com o resumo."""
    with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE ANÁLISE DE VÍDEO\n")
        f.write("=============================\n\n")
        f.write(f"Total de Frames Analisados: {frames}\n")
        f.write(f"Total de Anomalias (Movimentos Bruscos): {anomalias}\n\n")
        f.write("Contagem de Emoções Detectadas:\n")
        for emocao, qtd in emocoes.items():
            f.write(f"- {emocao}: {qtd}\n")
        f.write("\n=============================\n")
        f.write("Fim do Relatório.")

if __name__ == "__main__":
    processar_projeto_completo()