pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Download and Process Dataset') {
            steps {
                sh './scripts/download_and_process.sh'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'data/final_dataset.txt', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Pipeline zakończony sukcesem.'
        }
        failure {
            echo 'Pipeline zakończony niepowodzeniem.'
        }
    }
}