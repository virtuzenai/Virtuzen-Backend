const socketIo = require('socket.io');
let io;

const initSocket = (server) => {
  io = socketIo(server, { cors: { origin: '*' } });
  io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    socket.on('disconnect', () => console.log('Client disconnected:', socket.id));
  });
};

const emitWorkflowUpdate = (workflow) => {
  if (io) io.emit('workflowUpdate', workflow);
};

module.exports = { initSocket, emitWorkflowUpdate };
