const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Basic configuration to prevent EMFILE errors
config.watchFolders = [__dirname];

// Force watchman usage (more efficient than Node.js file watching)
config.resolver.useWatchman = true;

// Allow TypeScript files for Expo compatibility
config.resolver.sourceExts = ['js', 'jsx', 'ts', 'tsx', 'json'];

// Reduce platform extensions
config.resolver.platforms = ['ios', 'android'];

module.exports = config;
