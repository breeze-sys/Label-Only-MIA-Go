FROM golang:1.24-bookworm AS builder

WORKDIR /src

COPY go.mod ./
COPY main.go ./
COPY pkg ./pkg

RUN CGO_ENABLED=0 GOOS=linux go build -o /out/labelscan .

FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /out/labelscan /app/labelscan

CMD ["/app/labelscan"]
