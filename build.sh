#!/bin/bash
set -e

echo "🔧 Starting User Whisperer Platform build..."
echo "📁 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Build shared components first
echo "🏗️ Building shared components..."
cd shared
npm install
npm run build
cd ..

# Build the specific service
if [ -n "$SERVICE_NAME" ]; then
    echo "🎯 Building service: $SERVICE_NAME"
    cd services/$SERVICE_NAME
    npm install
    npm run build
    echo "✅ Service $SERVICE_NAME built successfully!"
else
    echo "❌ SERVICE_NAME environment variable not set"
    exit 1
fi

echo "🎉 Build completed successfully!"
