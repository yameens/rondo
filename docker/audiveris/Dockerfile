# Use a slim JDK 17 base
FROM eclipse-temurin:17-jre-jammy

# Create work dirs
WORKDIR /app
RUN mkdir -p /in /out

# Download Audiveris release
ADD https://github.com/Audiveris/audiveris/releases/download/5.6.2/Audiveris-5.6.2-linux-x86_64.zip /app/audiveris.zip

# Install dependencies and extract Audiveris
RUN apt-get update && apt-get install -y \
    unzip \
    tesseract-ocr \
    && unzip /app/audiveris.zip -d /app \
    && rm /app/audiveris.zip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JAVA_OPTS="-Xms512m -Xmx4g"
ENV AUDIVERIS_HOME=/app/Audiveris-5.6.2

# Simple entrypoint that forwards CLI args to Audiveris
ENTRYPOINT ["/bin/sh","-c","java $JAVA_OPTS -jar $AUDIVERIS_HOME/bin/Audiveris-5.6.2.jar \"$@\"", "--"]
