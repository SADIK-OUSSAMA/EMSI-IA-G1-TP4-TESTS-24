package ma.emsi.sadik.tests;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.sadik.assistant.Assistant;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test4 {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private static EmbeddingStore<TextSegment> ingestDocument(String resourceName,
                                                              DocumentParser parser,
                                                              EmbeddingModel embeddingModel) throws URISyntaxException {
        URL fileUrl = Test4.class.getResource(resourceName);
        Path path = Paths.get(fileUrl.toURI());
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        System.out.println("Ingestion terminée pour : " + resourceName);
        return embeddingStore;
    }

    public static void main(String[] args) throws URISyntaxException {

        configureLogger();
        String llmKey = System.getenv("GEMINI_API_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> ragStore = ingestDocument("/rag.pdf", new ApacheTikaDocumentParser(), embeddingModel);

        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.from(ragStore);

        // Classe interne locale, définie DANS la méthode main
        class RagQueryRouter implements QueryRouter {

            private final ChatModel chatModel;
            private final ContentRetriever contentRetriever;
            private final PromptTemplate promptTemplate;

            RagQueryRouter(ChatModel chatModel, ContentRetriever contentRetriever) {
                this.chatModel = chatModel;
                this.contentRetriever = contentRetriever;
                this.promptTemplate = PromptTemplate.from(
                        "Est-ce que la requête '{{query}}' porte sur l'IA ? Réponds seulement par 'oui', 'non', ou 'peut-être'."
                );
            }

            @Override
            public Collection<ContentRetriever> route(Query query) {
                Map<String, Object> variables = new HashMap<>();
                variables.put("query", query.text());
                String prompt = promptTemplate.apply(variables).text();

                String response = chatModel.chat(prompt);

                if (response.toLowerCase().contains("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter customQueryRouter = new RagQueryRouter(model, ragRetriever);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(customQueryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        String question;

        System.out.println("Posez votre question (tapez 'exit' pour quitter):");
        while (true) {
            question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) {
                break;
            }
            String reponse = assistant.chat(question);
            System.out.println("Réponse: " + reponse);
            System.out.println("\nPosez une autre question:");
        }

        scanner.close();
        System.out.println("Fin de la session.");
    }
}