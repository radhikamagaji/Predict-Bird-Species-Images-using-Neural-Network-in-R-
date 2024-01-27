PredictBirdSpecies_NN
================
Radhika
5/10/2022

``` r
library(ggplot2)
```

    ## Warning: package 'ggplot2' was built under R version 4.1.3

``` r
library(tidyverse)
```

    ## Warning: package 'tidyverse' was built under R version 4.1.3

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v tibble  3.1.6     v dplyr   1.0.8
    ## v tidyr   1.2.0     v stringr 1.4.0
    ## v readr   2.1.2     v forcats 0.5.1
    ## v purrr   0.3.4

    ## Warning: package 'tibble' was built under R version 4.1.3

    ## Warning: package 'tidyr' was built under R version 4.1.3

    ## Warning: package 'readr' was built under R version 4.1.3

    ## Warning: package 'purrr' was built under R version 4.1.3

    ## Warning: package 'dplyr' was built under R version 4.1.3

    ## Warning: package 'stringr' was built under R version 4.1.3

    ## Warning: package 'forcats' was built under R version 4.1.3

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(keras)
```

    ## Warning: package 'keras' was built under R version 4.1.3

``` r
library(tensorflow)
```

    ## Warning: package 'tensorflow' was built under R version 4.1.3

``` r
library(reticulate)
```

    ## Warning: package 'reticulate' was built under R version 4.1.3

``` r
label_list <- dir("E:\\Radhika\\SU_MSDS\\Spring22Quarter\\DATA5332_StatML2\\DATA5322_Assignments\\DATA_5332_HW3\\train_small3")
output_n <- length(label_list)
save(label_list, file="label_list.R")
```

``` r
width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels
```

``` r
path_train <- "E:\\Radhika\\SU_MSDS\\Spring22Quarter\\DATA5332_StatML2\\DATA5322_Assignments\\DATA_5332_HW3\\train_small3\\"
train_data_gen <- image_data_generator(rescale = 1/255, 
                  validation_split = .2)
```

    ## Loaded Tensorflow version 2.8.0

``` r
train_images <- flow_images_from_directory(path_train,
  train_data_gen,
  subset = 'training',
  target_size = target_size,
  class_mode = "categorical",
  shuffle=F,
  classes = label_list,
  seed = 2021)
```

``` r
table(train_images$classes)
```

    ## 
    ##  0  1  2 
    ## 80 80 80

``` r
validation_images <- flow_images_from_directory(path_train,
 train_data_gen, 
  subset = 'validation',
  target_size = target_size,
  class_mode = "categorical",
  classes = label_list,
  seed = 2021)
```

``` r
plot(as.raster(train_images[[1]][[1]][17,,,]))
```

![](PredictBirdSpecies_NN_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
mod_base <- application_xception(weights = 'imagenet', 
   include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 
```

``` r
model_function <- function(learning_rate = 0.001, 
  dropoutrate=0.2, n_dense=1024){
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(lr = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}
```

``` r
model <- model_function()
```

    ## Warning in backcompat_fix_rename_lr_to_learning_rate(...): the `lr` argument has
    ## been renamed to `learning_rate`.

``` r
model
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  xception (Functional)              (None, 7, 7, 2048)              20861480    
    ##                                                                                 
    ##  global_average_pooling2d (GlobalAv  (None, 2048)                   0           
    ##  eragePooling2D)                                                                
    ##                                                                                 
    ##  dense_1 (Dense)                    (None, 1024)                    2098176     
    ##                                                                                 
    ##  activation (Activation)            (None, 1024)                    0           
    ##                                                                                 
    ##  dropout (Dropout)                  (None, 1024)                    0           
    ##                                                                                 
    ##  dense (Dense)                      (None, 3)                       3075        
    ##                                                                                 
    ## ================================================================================
    ## Total params: 22,962,731
    ## Trainable params: 2,101,251
    ## Non-trainable params: 20,861,480
    ## ________________________________________________________________________________

``` r
batch_size <- 32
epochs <- 6
system.time(
  hist <- model %>% fit_generator(
    train_images,
    steps_per_epoch = train_images$n %/% batch_size, 
    epochs = epochs, 
    validation_data = validation_images,
    validation_steps = validation_images$n %/% batch_size,
    verbose = 2
  )
)
```

    ## Warning in fit_generator(., train_images, steps_per_epoch = train_images$n%/
    ## %batch_size, : `fit_generator` is deprecated. Use `fit` instead, it now accept
    ## generators.

    ##    user  system elapsed 
    ##  235.36   16.67   87.19

Evaluating and testing the model

``` r
path_test <- "E:\\Radhika\\SU_MSDS\\Spring22Quarter\\DATA5332_StatML2\\DATA5322_Assignments\\DATA_5332_HW3\\test_small3\\"

test_data_gen <- image_data_generator(rescale = 1/255)

test_images <- flow_images_from_directory(path_test,
   test_data_gen,
   target_size = target_size,
   class_mode = "categorical",
   classes = label_list,
   shuffle = F,
   seed = 2021)

model %>% evaluate_generator(test_images, 
                     steps = test_images$n)
```

    ## Warning in evaluate_generator(., test_images, steps = test_images$n):
    ## `evaluate_generator` is deprecated. Use `evaluate` instead, it now accept
    ## generators.

    ##       loss   accuracy 
    ## 0.02059142 0.99047619

Testing on a random image

``` r
test_image <- image_load("E:\\Radhika\\SU_MSDS\\Spring22Quarter\\DATA5332_StatML2\\DATA5322_Assignments\\DATA_5332_HW3\\barn_swallow_test.jpg",
                                  target_size = target_size)

x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred <- model %>% predict(x)
pred <- data.frame("Bird" = label_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:3,]
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred
```

    ##                 Bird  Probability
    ## 3     AMERICAN ROBIN 61.5968883 %
    ## 2 AMERICAN GOLDFINCH 37.4606550 %
    ## 1      AMERICAN CROW  0.9424561 %

``` r
# Plot confusion matrix
#onfMat <- confusionMatrix(data = factor(predClass, levels = speciesClass),
 #                          reference = factor(trueClass, levels = speciesClass))
```
