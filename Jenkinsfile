pipeline {
    agent any

    parameters {
        string(name: 'CUTOFF', defaultValue: '10', description: 'Liczba przykładów')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Generate Dataset') {
            steps {
                sh 'chmod +x scripts/download_and_process.sh'
                sh "./scripts/download_and_process.sh ${params.CUTOFF}"
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'data/dataset.txt', fingerprint: true
            }
        }
    }
}