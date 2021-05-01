name := "image-classification"

version := "0.1"

scalaVersion := "2.12.13"

resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.1"

// Using Deep Java Learn
libraryDependencies += "ai.djl" % "api" % "0.10.0"
libraryDependencies += "ai.djl" % "repository" % "0.4.1"

// Using MXNet Engine
libraryDependencies += "ai.djl.mxnet" % "mxnet-model-zoo" % "0.10.0"
libraryDependencies += "ai.djl.mxnet" % "mxnet-native-auto" % "1.8.0"
