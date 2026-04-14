pipeline {
    agent any

    parameters {
        string(name: 'EPOCHS', defaultValue: '30', description: 'Liczba epok')
    }

    stages {

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("diabetes-model")
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    docker.image("diabetes-model").inside {
                        sh """
                        echo "Training with EPOCHS=${EPOCHS}"
                        uv run python train.py --epochs ${EPOCHS}
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: 'model.pth', fingerprint: true
        }
    }
}