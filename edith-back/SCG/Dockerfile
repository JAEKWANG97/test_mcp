# 빌드 단계
#FROM gradle:7.6.1-jdk17 AS builder
FROM amazoncorretto:17

# 작업 디렉토리 설정
WORKDIR /home/gradle/project

# 소스 코드 복사
COPY . .

# 빌드 및 문서 생성
RUN gradle clean build -x test

# 실행 단계
FROM openjdk:17

# 타임존 설정
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone

# JAR 파일 복사
# COPY --from=builder /home/gradle/project/build/libs/SCG-0.0.1-SNAPSHOT.jar /SCG.jar
ARG JAR_FILE
COPY ${JAR_FILE} scg.jar

# 실행
ENTRYPOINT ["java", "-jar","-Dspring.profiles.active=prod", "-Dserver.port=8080", "-Duser.timezone=Asia/Seoul", "/scg.jar"]
