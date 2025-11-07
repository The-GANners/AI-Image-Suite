import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { 
  signInWithPopup, 
  signOut, 
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  updateProfile
} from 'firebase/auth';
import { STORAGE_KEYS, AUTH_MESSAGES } from '../utils/constants';
import { auth, googleProvider } from '../utils/firebase';

export const AuthContext = createContext();

export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState('');
  const [showLoginModal, setShowLoginModal] = useState(false);
  const navigate = useNavigate();

  const handleLogout = useCallback((showToast = true, reason = '') => {
    signOut(auth).catch((error) => {
      console.error('Firebase sign-out error:', error);
    });

    localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER_INFO);
    localStorage.removeItem('last_activity_time');

    setCurrentUser(null);
    setAuthError('');

    if (showToast) {
      if (reason === 'timeout') {
        toast.error('Session expired due to inactivity. Please login again.');
      } else {
        toast.success(AUTH_MESSAGES.LOGOUT_SUCCESS);
      }
    }

    navigate('/login');
  }, [navigate]);

  const verifyToken = useCallback(async (token) => {
    try {
      const response = await fetch('/api/auth/verify', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        handleLogout(false);
        toast.error(AUTH_MESSAGES.SESSION_EXPIRED);
      }
    } catch (error) {
      console.error('Token verification error:', error);
    }
  }, [handleLogout]);

  useEffect(() => {
    // Check for stored token and user info on initial load
    const storedToken = localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
    const storedUserInfo = localStorage.getItem(STORAGE_KEYS.USER_INFO);
    
    if (storedToken && storedUserInfo) {
      try {
        const userInfo = JSON.parse(storedUserInfo);
        setCurrentUser(userInfo);
        
        // Verify token validity with server
        verifyToken(storedToken);
      } catch (error) {
        // Invalid stored data, clear it
        handleLogout(false);
      }
    }
    
    setLoading(false);
  }, [handleLogout, verifyToken]);

  // 30-minute session timeout effect
  useEffect(() => {
    if (!currentUser) return;

    const TIMEOUT_DURATION = 30 * 60 * 1000; // 30 minutes in milliseconds
    let timeoutId;

    const updateLastActivity = () => {
      localStorage.setItem('last_activity_time', Date.now().toString());
    };

    const checkInactivity = () => {
      const lastActivity = parseInt(localStorage.getItem('last_activity_time') || Date.now());
      const currentTime = Date.now();
      const inactiveTime = currentTime - lastActivity;

      if (inactiveTime >= TIMEOUT_DURATION) {
        handleLogout(true, 'timeout');
      }
    };

    // Initialize last activity time
    updateLastActivity();

    // Set up interval to check inactivity every minute
    const intervalId = setInterval(checkInactivity, 60000);

    // Track user activity
    const activityEvents = ['mousedown', 'keydown', 'scroll', 'touchstart', 'click'];
    
    const handleActivity = () => {
      updateLastActivity();
      // Reset timeout on activity
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        handleLogout(true, 'timeout');
      }, TIMEOUT_DURATION);
    };

    // Add event listeners
    activityEvents.forEach(event => {
      window.addEventListener(event, handleActivity);
    });

    // Set initial timeout
    timeoutId = setTimeout(() => {
      handleLogout(true, 'timeout');
    }, TIMEOUT_DURATION);

    // Cleanup
    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
      activityEvents.forEach(event => {
        window.removeEventListener(event, handleActivity);
      });
    };
  }, [currentUser, handleLogout]);

  const handleLogin = async (email, password) => {
    try {
      setAuthError('');
      setLoading(true);
      
      // Sign in with Firebase first
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const firebaseUser = userCredential.user;
      const firebaseToken = await firebaseUser.getIdToken();
      
      // Send Firebase token to backend
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ firebaseToken }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || AUTH_MESSAGES.LOGIN_ERROR);
      }

      // Store token and user info
      localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, data.token);
      localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(data.user));
      
      // Update state
      setCurrentUser(data.user);
      toast.success(AUTH_MESSAGES.LOGIN_SUCCESS);
      navigate('/');
      
      return true;
    } catch (error) {
      console.error('Login error:', error);
      
      // Handle Firebase-specific errors
      let message = error.message;
      if (error.code === 'auth/user-not-found') {
        message = 'No account found with this email';
      } else if (error.code === 'auth/wrong-password') {
        message = 'Invalid email or password';
      } else if (error.code === 'auth/invalid-email') {
        message = AUTH_MESSAGES.INVALID_EMAIL;
      } else if (error.code === 'auth/too-many-requests') {
        message = 'Too many failed attempts. Please try again later';
      } else if (error.code === 'auth/invalid-credential') {
        message = 'Invalid email or password';
      }
      
      setAuthError(message);
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      setAuthError('');
      setLoading(true);

      const result = await signInWithPopup(auth, googleProvider);
      const firebaseUser = result.user;
      const firebaseToken = await firebaseUser.getIdToken();

      const response = await fetch('/api/auth/firebase-login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          firebaseToken,
          user: {
            displayName: firebaseUser.displayName,
            email: firebaseUser.email,
            photoURL: firebaseUser.photoURL,
          },
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || AUTH_MESSAGES.GOOGLE_SIGNIN_ERROR);
      }

      localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, data.token);
      localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(data.user));

      setCurrentUser(data.user);
      toast.success(AUTH_MESSAGES.LOGIN_SUCCESS);
      navigate('/');

      return true;
    } catch (error) {
      console.error('Google login error:', error);

      let message = error?.message || AUTH_MESSAGES.GOOGLE_SIGNIN_ERROR;
      if (error?.code === 'auth/popup-closed-by-user') {
        message = AUTH_MESSAGES.GOOGLE_SIGNIN_CANCELLED;
      }

      setAuthError(message);
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (name, email, password) => {
    try {
      setAuthError('');
      setLoading(true);
      
      // Create user in Firebase first
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const firebaseUser = userCredential.user;
      
      // Update Firebase profile with name
      await updateProfile(firebaseUser, {
        displayName: name
      });
      
      // Get Firebase token
      const firebaseToken = await firebaseUser.getIdToken();
      
      // Send to backend to create user record in our database
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          firebaseToken,
          name
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        // If backend fails, we should probably delete the Firebase user
        // but for now, just throw the error
        throw new Error(data.message || AUTH_MESSAGES.SIGNUP_ERROR);
      }

      // Store token and user info
      localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, data.token);
      localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(data.user));
      
      // Update state
      setCurrentUser(data.user);
      toast.success(AUTH_MESSAGES.SIGNUP_SUCCESS);
      navigate('/');
      
      return true;
    } catch (error) {
      console.error('Signup error:', error);
      
      // Handle Firebase-specific errors
      let message = error.message;
      if (error.code === 'auth/email-already-in-use') {
        message = 'This email is already registered';
      } else if (error.code === 'auth/invalid-email') {
        message = AUTH_MESSAGES.INVALID_EMAIL;
      } else if (error.code === 'auth/operation-not-allowed') {
        message = 'Email/password authentication is not enabled';
      } else if (error.code === 'auth/weak-password') {
        message = 'Password should be at least 6 characters';
      }
      
      setAuthError(message);
      toast.error(message);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const openLoginModal = useCallback(() => {
    setShowLoginModal(true);
  }, []);

  const closeLoginModal = useCallback(() => {
    setShowLoginModal(false);
    setAuthError('');
  }, []);

  const requireAuth = useCallback((callback) => {
    if (currentUser) {
      callback();
    } else {
      openLoginModal();
    }
  }, [currentUser, openLoginModal]);

  const value = {
    currentUser,
    isAuthenticated: !!currentUser,
    loading,
    error: authError,
    login: handleLogin,
    loginWithGoogle: handleGoogleLogin,
    signup: handleSignup,
    logout: handleLogout,
    showLoginModal,
    openLoginModal,
    closeLoginModal,
    requireAuth,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
