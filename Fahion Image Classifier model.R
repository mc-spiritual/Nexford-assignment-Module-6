library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(caret)       # for confusion matrix
#library(pheatmap)    # for heatmap visualization

ApparelClassifier <- R6::R6Class(
  "ApparelClassifier",
  
  public = list(
    num_classes = 10,
    input_shape = c(28, 28, 1),
    class_names = c(
      "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
      "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ),
    
    model = NULL,
    history = NULL,
    x_train = NULL,
    y_train = NULL,
    x_test = NULL,
    y_test_original = NULL,
    y_test = NULL,
    
    load_data = function() {
      cat("Fetching dataset...\n")
      dataset <- dataset_fashion_mnist()
      self$x_train <- dataset$train$x
      self$y_train <- dataset$train$y
      self$x_test <- dataset$test$x
      self$y_test_original <- dataset$test$y
      self$y_test <- self$y_test_original
      cat("Train samples:", dim(self$x_train), "\n")
      cat("Test samples:", dim(self$x_test), "\n")
    },
    
    preprocess_data = function() {
      cat("Preparing images for training...\n")
      self$x_train <- self$x_train / 255
      self$x_test <- self$x_test / 255
      self$x_train <- array_reshape(self$x_train, c(nrow(self$x_train), 28, 28, 1))
      self$x_test <- array_reshape(self$x_test, c(nrow(self$x_test), 28, 28, 1))
      self$y_train <- to_categorical(self$y_train, self$num_classes)
    },
    
    create_model = function() {
      cat("Initializing model architecture...\n")
      self$model <- keras_model_sequential() %>%
        layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                      kernel_initializer = "he_normal", input_shape = self$input_shape) %>%
        layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
        layer_max_pooling_2d(pool_size = c(2,2)) %>%
        layer_dropout(rate = 0.25) %>%
        layer_flatten() %>%
        layer_dense(units = 128, activation = "relu", kernel_initializer = "he_normal") %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(units = self$num_classes, activation = "softmax")
      
      summary(self$model)
    },
    
    setup_model = function() {
      cat("Compiling model...\n")
      self$model %>% compile(
        loss = "categorical_crossentropy",
        optimizer = optimizer_adam(learning_rate = 0.001),
        metrics = c("accuracy", metric_precision(name = "precision"), metric_recall(name = "recall"))
      )
    },
    
    run_training = function(batch_size = 128, total_epochs = 15) {
      cat(sprintf("Training for %d epochs...\n", total_epochs))
      self$history <- self$model %>% fit(
        self$x_train, self$y_train,
        batch_size = batch_size,
        epochs = total_epochs,
        validation_split = 0.1,
        verbose = 1
      )
    },
    
    test_model = function() {
      cat("Evaluating model...\n")
      metrics <- self$model %>% evaluate(
        self$x_test, to_categorical(self$y_test), verbose = 0
      )
      acc <- metrics[[2]]
      prec <- metrics[[3]]
      rec <- metrics[[4]]
      f1 <- 2 * (prec * rec) / (prec + rec)
      cat(sprintf("Accuracy: %.4f\nPrecision: %.4f\nRecall: %.4f\nF1-score: %.4f\n", acc, prec, rec, f1))
      
      # Predictions
      preds <- self$model %>% predict(self$x_test)
      predicted_labels <- apply(preds, 1, which.max) - 1  # R is 1-indexed
      self$plot_confusion(self$y_test, predicted_labels)
      self$plot_training()
      
      # Classification Report (summary)
      cat("\nClassification Report:\n")
      print(confusionMatrix(factor(predicted_labels), factor(self$y_test)))
    },
    
    plot_training = function() {
      if (is.null(self$history)) {
        cat("No training history.\n")
        return()
      }
      
      df <- data.frame(
        epoch = seq_along(self$history$metrics$accuracy),
        accuracy = self$history$metrics$accuracy,
        val_accuracy = self$history$metrics$val_accuracy,
        loss = self$history$metrics$loss,
        val_loss = self$history$metrics$val_loss
      )
      
      p1 <- ggplot(df, aes(x = epoch)) +
        geom_line(aes(y = accuracy, color = "Train")) +
        geom_line(aes(y = val_accuracy, color = "Validation")) +
        labs(title = "Model Accuracy", y = "Accuracy") +
        theme_minimal()
      
      p2 <- ggplot(df, aes(x = epoch)) +
        geom_line(aes(y = loss, color = "Train")) +
        geom_line(aes(y = val_loss, color = "Validation")) +
        labs(title = "Model Loss", y = "Loss") +
        theme_minimal()
      
      gridExtra::grid.arrange(p1, p2, nrow = 1)
    },
    
    plot_confusion = function(true, predicted) {
      cm <- table(True = true, Predicted = predicted)
      pheatmap(as.matrix(cm), cluster_rows = FALSE, cluster_cols = FALSE,
               display_numbers = TRUE, fontsize_number = 10,
               main = "Confusion Matrix", angle_col = 45)
    },
    
    predict_images = function(indexes = c(1, 12, 42, 100)) {
      par(mfrow = c(1, length(indexes)))
      for (i in indexes) {
        img <- self$x_test[i,,,drop=FALSE]
        probs <- self$model %>% predict(img)
        pred_class <- which.max(probs) - 1
        true_class <- self$y_test[i]
        
        image(t(apply(self$x_test[i,,,1], 2, rev)), col = gray.colors(255), axes = FALSE,
              main = sprintf("Pred: %s\nTrue: %s", 
                             self$class_names[pred_class + 1], 
                             self$class_names[true_class + 1]),
              xlab = "", ylab = "")
        
        cat(sprintf("\nImage %d:\n", i))
        cat(sprintf("True: %s (%d)\n", self$class_names[true_class + 1], true_class))
        cat(sprintf("Pred: %s (%d)\n", self$class_names[pred_class + 1], pred_class))
        cat(sprintf("Confidence: %.2f%%\n", max(probs) * 100))
      }
    },
    
    explain_performance = function() {
      cat("\n=== Performance Explanation ===\n")
      cat("1. Accuracy - general prediction correctness.\n")
      cat("2. Precision - among predicted labels, how many were correct.\n")
      cat("3. Recall - among actual labels, how many were found correctly.\n")
      cat("4. F1 - balance between precision and recall.\n")
      cat("\nInspect confusion matrix to see specific misclassification patterns.\n")
    },
    
    persist_model = function(path = "fashion_mnist_model.h5") {
      save_model_hdf5(self$model, path)
      cat(sprintf("Model stored at: %s\n", path))
    }
  )
)

# using the classifier
clf <- ApparelClassifier$new()
clf$load_data()
clf$preprocess_data()
clf$create_model()
clf$setup_model()
clf$run_training(total_epochs = 15)
clf$test_model()
clf$explain_performance()
clf$predict_images(c(1, 12, 42, 100))
clf$persist_model("fashion_mnist_model.h5")
