package com.github.denisnovac.app

import ai.djl.Application
import ai.djl.modality.Classifications
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.{BufferedImageFactory, Image}
import ai.djl.repository.zoo.{Criteria, ModelZoo}
import ai.djl.training.util.ProgressBar
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main extends App {

  /* Конфигурация Spark */

  val conf = new SparkConf()
    .setAppName("Image Classification Test")
    .setMaster("local[*]")
    .setExecutorEnv("MXNET_ENGINE_TYPE", "NaiveEngine") // NaiveEngine нужен для мультитрединга в MXNet

  val sc = new SparkContext(conf)

  /* Данные для инпута */

  // Распределение всех файлов между partition-ами Spark равномерно
  val path       = getClass.getClassLoader.getResource("images")
  val partitions = sc.binaryFiles(s"${path.toString}/*")

  /* Networks: https://github.com/deepjavalibrary/djl/tree/master/mxnet/mxnet-model-zoo/src/test/resources/mlrepo/model/cv/image_classification/ai/djl/mxnet */
  // image classification: позволяет извлечь класс объекта: спортивная машина, немецкая овчарка
  val alexnet = Criteria.builder
    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
    .setTypes(classOf[Image], classOf[Classifications])
    .optFilter("dataset", "imagenet")
    .optProgress(new ProgressBar)
    .build

  val vgg16 = Criteria.builder
    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
    .setTypes(classOf[Image], classOf[Classifications])
    .optFilter("layers", "16")
    .optFilter("dataset", "imagenet")
    .optProgress(new ProgressBar)
    .build

  val xception = Criteria.builder
    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
    .setTypes(classOf[Image], classOf[Classifications])
    .optFilter("flavor", "65")
    .optFilter("dataset", "imagenet")
    .optProgress(new ProgressBar)
    .build

  // object detection: делит на типы объектов - персона, машина
  val ssd512 = Criteria.builder
    .optApplication(Application.CV.OBJECT_DETECTION)
    .setTypes(classOf[Image], classOf[DetectedObjects])
    .optFilter("size", "512")
    .optFilter("backbone", "resnet50")
    .optFilter("flavor", "v1")
    .optFilter("dataset", "voc")
    .optProgress(new ProgressBar)
    .build

  val yolo = Criteria.builder
    .optApplication(Application.CV.OBJECT_DETECTION)
    .setTypes(classOf[Image], classOf[DetectedObjects])
    .optFilter("imageSize", "320")
    .optFilter("backbone", "darknet53")
    .optFilter("version", "3")
    .optFilter("dataset", "voc")
    .optProgress(new ProgressBar)
    .build

  /* Spark Job */

  val result: RDD[String] = partitions.mapPartitions { partition =>
    val model     = ModelZoo.loadModel(yolo)
    println(model.getName)
    val predictor = model.newPredictor()

    // classification
    partition.map { streamData =>
      val name       = streamData._1
      val img        = new BufferedImageFactory().fromInputStream(streamData._2.open())
      val predictRes = predictor.predict(img).toString

      s"$name:\n$predictRes"
    }
  }

  result.collect().foreach(println)
}
