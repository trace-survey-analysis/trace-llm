node {

    stage('Clone repository') {
        checkout scm
    }

    stage('Semantic Release') {
            withCredentials([usernamePassword(credentialsId: 'github-pat', usernameVariable: 'GITHUB_USERNAME', passwordVariable: 'GITHUB_TOKEN')]) {
                script {
                    echo "Running semantic-release"
                    sh "npx semantic-release"
                }
            }    
        }
    stage('Build and Push multi-platform image') {
        withCredentials([usernamePassword(credentialsId: 'docker-pat', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_TOKEN')]) {
            script{
                // Get the latest version
                def LATEST_TAG = sh(script: "git describe --tags --abbrev=0", returnStdout: true).trim()
                // Login to Docker
                sh """
                    docker login -u ${DOCKER_USERNAME} -p ${DOCKER_TOKEN}
                """
                
                // Setup buildx
                sh """
                    docker buildx create --use --name builder || docker buildx use builder
                    docker buildx inspect --bootstrap
                """
                
                // Build and push multi-platform image
                sh """
                    docker buildx build \\
                        --platform linux/amd64,linux/arm64 \\
                        -t roarceus/trace-llm:${LATEST_TAG} \\
                        -t roarceus/trace-llm:latest \\
                        --push . \\
                        --progress=plain
                """
            }
        }
    }
}