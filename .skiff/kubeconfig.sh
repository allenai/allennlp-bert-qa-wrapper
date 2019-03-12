#!/usr/bin/env bash
set -e

#
# Script that generates the appropriate kubernetes manifest given the
# provided parameters.
#
# This is intentionally simple. If we end up doing complex things, we should
# likely embrace jsonnet or another community accepted solution for templated
# kubernetes configuration files.
#

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

usage() {
  echo ""
  echo "Usage:"
  echo "  ./kubeconfig.sh ENV IMAGE"
  echo ""
  echo "  ENV   the environment identfier that indicates what configuration file to include, i.e. app.dev.config.sh"
  echo "  IMAGE the docker image name to deploy"
  echo ""
}

# Validate arguments
ENV=$1
if [[ -z "$ENV" ]]; then
  (>&2 echo "Error: no ENV specified.")
  usage
  exit 1
fi

IMAGE=$2
if [[ -z "$IMAGE" ]]; then
  (>&2 echo "Error: no IMAGE specified.")
  usage
  exit 1
fi

# Validate required environment variables
if [[ -z "$REPO_NAME" ]]; then
  (>&2 echo "Error: the REPO_NAME environment variable must be set.")
  usage
  exit 1
fi

# Load the environment specific config
source "$dir/config-$ENV.sh"

# Verify the config
if [[ -z "$DOMAIN" ]]; then
  (>&2 echo "Error: Invalid configuration for environment $ENV, the DOMAIN config parameter isn't set.")
  exit 1
fi

if [[ -z "$HTTP_PORT" ]]; then
  (>&2 echo "Error: Invalid configuration for environment $ENV, the HTTP_PORT config parameter isn't set.")
  exit 1
fi

# Strip the "github_allenai_" prefix from the REPO_NAME variable set by Google
# when they mirror or repos.
APP_NAME=${REPO_NAME#"github_allenai_"}

# For certain resources we need to use a name that includes the environment
# since there might
FULLY_QUALIFIED_APP_NAME="$APP_NAME-$ENV"

# Generate Kubernetes config
config=$(cat <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $APP_NAME
---
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: $FULLY_QUALIFIED_APP_NAME
  namespace: $APP_NAME
  labels:
    app: $APP_NAME
    env: $ENV
  annotations:
    certmanager.k8s.io/cluster-issuer: letsencrypt-prod
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/enable-cors: "false"
spec:
  tls:
    - secretName: $FULLY_QUALIFIED_APP_NAME-tls
      hosts:
      - $DOMAIN
  rules:
  - host: $DOMAIN
    http:
      paths:
      - backend:
          serviceName: $FULLY_QUALIFIED_APP_NAME
          servicePort: 80
---
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    app: $APP_NAME
    env: $ENV
  name: $FULLY_QUALIFIED_APP_NAME
  namespace: $APP_NAME
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: $APP_NAME
        env: $ENV
    spec:
      containers:
        - name: $FULLY_QUALIFIED_APP_NAME
          image: $IMAGE
          # Checks whether a newly started pod is ready to receive traffic.
          # After 6 failed checks spaced by 10 seconds, the pod will be marked Unready.
          # See https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-probes
          readinessProbe:
            failureThreshold: 6
            periodSeconds: 10
            httpGet:
              path: /
              port: $HTTP_PORT
              scheme: HTTP
            initialDelaySeconds: 1
          # Checks whehter a pod is still alive and can continue to receive traffic.
          # After 6 failed checks spaced by 10 seconds, the pod will be killed and restarted.
          # See https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-probes
          livenessProbe:
            failureThreshold: 6
            periodSeconds: 50
            httpGet:
              path: /
              port: $HTTP_PORT
              scheme: HTTP
            initialDelaySeconds: 60
          resources:
            requests:
              cpu: "1"
              memory: 1000Mi
---
apiVersion: v1
kind: Service
metadata:
  name: $FULLY_QUALIFIED_APP_NAME
  namespace: $APP_NAME
  labels:
    app: $APP_NAME
    env: $ENV
spec:
  selector:
    app: $APP_NAME
    env: $ENV
  ports:
    - port: 80
      targetPort: $HTTP_PORT
      name: http
EOF
)

# Write it to a file
outfile=$dir/../kubeconfig.yaml
echo "$config" > "$outfile"

echo "generated $outfile"
