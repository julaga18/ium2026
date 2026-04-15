pipeline {
    agent {
        dockerfile true
    }

    parameters {
        string(name: 'EPOCHS', defaultValue: '30', description: 'Liczba epok')
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Prepare workspace') {
            steps {
                sh "mkdir -p data"
            }
        }

        stage('Train Model') {
            steps {
                sh """
                echo "TRAINING | EPOCHS=${params.EPOCHS}"
                python train.py --epochs ${params.EPOCHS}
                """
            }
        }

        stage('Predict') {
            steps {
                sh """
                echo "PREDICTION"
                python predict.py
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: '*.csv, *.pkl, *.pth, data/*.csv, data/*.pth', fingerprint: true
            }
        }
    }

    post {
        success {
            echo "Pipeline finished successfully"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}