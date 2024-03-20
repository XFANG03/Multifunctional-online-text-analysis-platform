-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: fakenews
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `feedback`
--
CREATE DATABASE fakenews;
USE fakenews;

DROP TABLE IF EXISTS `feedback`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `feedback` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) DEFAULT NULL,
  `message` varchar(1000) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `rate` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `feedback`
--

LOCK TABLES `feedback` WRITE;
/*!40000 ALTER TABLE `feedback` DISABLE KEYS */;
INSERT INTO `feedback` VALUES (5,'lily','this webside is super great, soving my problem and question. Help me a lot!','123@qq.com',4);
/*!40000 ALTER TABLE `feedback` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `text_record`
--

DROP TABLE IF EXISTS `text_record`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `text_record` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text_content` text NOT NULL,
  `record_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `text_length` int DEFAULT NULL,
  `username` varchar(50) NOT NULL,
  `mode` varchar(20) DEFAULT NULL,
  `robot_answer` varchar(1024) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `new_text_record_ibfk_1` (`username`),
  CONSTRAINT `new_text_record_ibfk_1` FOREIGN KEY (`username`) REFERENCES `user` (`username`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=378 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `text_record`
--

LOCK TABLES `text_record` WRITE;
/*!40000 ALTER TABLE `text_record` DISABLE KEYS */;
INSERT INTO `text_record` VALUES (370,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:30:48',405,'Flora','2','The keywordsThe key words in this passage are: [evening,glow,silence,beauty,sound,horizon,perfect,breeze,warm,scene]'),(371,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:30:54',405,'Flora','3','{\n  \"summary\": \"\\\"the sun was setting behind the distant mountains, casting a warm orange glow across the horizon.\\\" \\\"a gentle breeze rustled the leaves, creating a soothing sound that filled the air.\\\" \\\"it was a perfect moment to reflect and appreciate the beauty of nature.\\\"\"\n}\n'),(372,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:37:50',405,'Flora','3','The summary of this news is: \"the sun was setting behind the distant mountains, casting a warm orange glow across the horizon.\" \"a gentle breeze rustled the leaves, creating a soothing sound that filled the air.\" \"it was a perfect moment to reflect and appreciate the beauty of nature.\"'),(373,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:40:05',405,'Flora','4','The sentiment of this news is: joy (probability: 0.4621)'),(374,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:43:42',405,'Flora','6','This piece of news is a: Fake News'),(375,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:45:25',405,'Flora','5','The topic of this news is: environment (probability: 0.0723)'),(376,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-16 05:48:21',405,'Flora','5','The topic of this news is: environment (probability: 0.0723)'),(377,'\"The sun was setting behind the distant mountains, casting a warm orange glow across the horizon. A gentle breeze rustled the leaves of the trees, creating a soothing sound that filled the air. As the day drew to a close, the chirping of birds gradually subsided, giving way to the peaceful silence of the evening. It was a tranquil scene, a perfect moment to reflect and appreciate the beauty of nature.\"','2023-11-23 12:45:28',405,'Flora','4','The sentiment of this news is: joy (probability: 0.4621)');
/*!40000 ALTER TABLE `text_record` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user` (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(250) NOT NULL,
  `phone_number` varchar(20) NOT NULL,
  `gender` varchar(10) DEFAULT NULL,
  `age` int DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `birthdate` date DEFAULT NULL,
  `contact_address` varchar(100) DEFAULT NULL,
  `comment` varchar(255) DEFAULT NULL,
  `image_url` varchar(255) DEFAULT NULL,
  `address` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `phone_number` (`phone_number`),
  UNIQUE KEY `uniq_username` (`username`),
  KEY `idx_username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=19 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user`
--

LOCK TABLES `user` WRITE;
/*!40000 ALTER TABLE `user` DISABLE KEYS */;
INSERT INTO `user` VALUES (17,'123','96cae35ce8a9b0244178bf28e4966c2ce1b8385723a96a6b838858cdd6ca0a1e','12312312312','Female',3,'123@qq.com','2020-11-17','GuangDong','school!','/user/users/download?name=92b49693-d691-4ff7-ad3d-477e510d2c6a.jpg','guangzhou'),(18,'Flora','92925488b28ab12584ac8fcaa8a27a0f497b2c62940c8f4fbc8ef19ebc87c43e','19860318728','Female',19,'888@qq.com','2004-02-21','GuangDong','work work work !','/user/users/download?name=d73df3c3-90dd-4ef5-b43c-2a28ddd1f72d.jpg','GuangDong');
/*!40000 ALTER TABLE `user` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-12-01 13:43:38
